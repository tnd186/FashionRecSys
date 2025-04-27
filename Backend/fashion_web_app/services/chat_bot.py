# Standard library
import json
import os
import re
import textwrap
from typing import Any, Dict, Optional, Tuple, List

# Third-party libraries
import torch
from django.forms.models import model_to_dict
from peft import PeftModel
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, Pipeline

# LangChain and extensions
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# ORM Models
from fashion_web_app.models import Product


class ChatBot:
    def __init__(self, model_path_tsq: Optional[str] = None, model_path_tc: Optional[str] = None) -> None:
        """
        Initialize the ChatBot, including the question shortening and answer generation pipelines,
        a reranker, and embeddings to support the Q&A system.

        Args:
            model_path_tsq: Path to the question shortening model.
            model_path_tc: Path to the chatbot model.

        Returns:
            None
        """
        print("\nInitializing chatbot")

        # Set the path to the models
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path_tsq is None:
            model_path_tsq = os.path.join(base_dir, "weights", "TinyLlama_Shorten_Question")
        if model_path_tc is None:
            model_path_tc = os.path.join(base_dir, "weights", "TinyLlama_Chatbot")

        # Load the question shortening pipeline
        gen_config_tsq = GenerationConfig(
            max_new_tokens=50,
            temperature=0.3,
            top_p=0.8,
            num_return_sequences=1,
        )
        self.pipe_tsq = self.load_pipeline(model_path_tsq, gen_config_tsq)

        # Load the answer generation pipeline
        gen_config_tc = GenerationConfig(
            max_new_tokens=250,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
        )
        self.pipe_tc = self.load_pipeline(model_path_tc, gen_config_tc)

        # Initialize the reranker using the CrossEncoder model
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Initialize embeddings for the search system
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create the QA chain
        self.qa_chain = self.create_qa_chain()


    def __call__(self, user_message: str) -> str: 
        """
        Call the QA chain to generate a response to the user's message.

        Args:
            user_message: The user's message.

        Returns:
            extracted_text: The generated model response.
        """

        # Retrieve context from MongoDB based on the user's message
        mongo_context = self.get_mongo_context(user_message)

        # Invoke the QA chain with the input parameters
        response = self.qa_chain.invoke({
            "question": user_message,
            "mongo_context": mongo_context,
            "chat_history": [],
            "document_context": []
        })

        # Extract the answer text from the model's response
        match = re.search(r"<\|assistant\|>\s*(.*)", response["answer"], re.DOTALL)
        extracted_text: str = (
            match.group(1)
            .strip()
            .split("\n\n")[0]
            .replace("\\", "/")
        )
        
        return extracted_text


    def get_mongo_retriever(self) -> Retriever: 
        """
        Retrieve product data from MongoDB and initialize a retriever for those products.

        Returns:
            mongo_retriever: A retriever object for querying products from MongoDB.
        """

        # Determine the directory for storing the vector database from MongoDB
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        persist_dir = os.path.join(base_dir, "chroma_db")

        # Query products from MongoDB
        products = Product.objects.all()
        print(f"\nRetrieved {len(products)} products from MongoDB")

        # Convert each product into a Document
        docs = []
        for product in products:
            product_dict = model_to_dict(product)
            # Filter out unnecessary fields from metadata
            metadata = {
                key: value
                for key, value in product_dict.items()
                if key not in ("id", "positive_image")
            }
            docs.append(
                Document(
                    page_content=product_dict.get("product_name", ""),
                    metadata=metadata,
                )
            )

        # Delete the old product collection from previous queries
        chroma_db = Chroma(
            collection_name="fashion_products",
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
        )
        chroma_db.delete_collection()

        # Create a new collection from the current products
        chroma_db = Chroma.from_documents(
            docs,
            self.embeddings,
            collection_name="fashion_products",
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

        # Convert the Chroma vector database into a retriever
        mongo_retriever = chroma_db.as_retriever()
        print("\nMongoDB retriever created successfully")

        return mongo_retriever


    def get_mongo_context(self, query: str) -> str:
        """
        Shorten the query to extract product content for providing context in the prompt.

        Args:
            query: The message from the user.

        Returns:
            final_context: A string containing product-related context retrieved from MongoDB.
        """
        # Normalize the query
        query = query.strip()
        if not query.endswith("?"):
            query += "?"

        # Create the prompt for query shortening
        prompt = textwrap.dedent(f"""
            <|system|>
            Bạn là trợ lý AI chuyển đổi câu hỏi sang dạng ngắn gọn hơn.</s>
            <|user|>
            {query}</s>
            <|assistant|>
            """
        )
        output = self.pipe_tsq(prompt)

        # Extract the shortened query from the model output
        match = re.search(r"<\|assistant\|>\s*(.*)", output[0]['generated_text'], re.DOTALL)
        new_query = match.group(1).strip().splitlines()[0]
        print(f"\nQuery after shortening: {new_query}")

        # Determine k and filter conditions for the MongoDB query
        k, filter_conditions = self.get_k_filter_conditions(new_query)
        print(f"\nDetermined k: {k}, filter_conditions: {filter_conditions}")

        # Initialize the Mongo retriever and perform vector search
        mongo_retriever = self.get_mongo_retriever()
        if filter_conditions is None:
            results = mongo_retriever.vectorstore.similarity_search_with_score(new_query, k=50)
        else:
            results = mongo_retriever.vectorstore.similarity_search_with_score(new_query, k=50, filter=filter_conditions)
        print("\nInitial product list retrieved successfully")

        # Perform reranking and filter based on score threshold
        rerank_input = [(new_query, doc.page_content) for doc, _ in results]
        scores = self.reranker.predict(rerank_input)
        reranked_results = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        filtered_results = [item for item in reranked_results if item[1] > -8]
        final_results = [x[0][0].metadata for x in filtered_results[:k]]

        # Map metadata keys to formatted strings
        mapping_dict = {
            "category": "- **Danh mục:**",
            "product_name": "- **Tên:**",
            "price": "- **Giá:**",
            "stock": "- **Tồn kho:**",
            "positive_url": "- **Đường dẫn hình ảnh:**"
        }
        field_order = ["category", "positive_url", "price", "product_name", "stock"]

        # Iterate through all documents and rename keys
        mongo_contexts = []
        for metadata in final_results:
            formatted_metadata = "\n".join([f"{mapping_dict.get(key)} {metadata.get(key, '')}" for key in field_order if key in metadata])
            mongo_contexts.append(formatted_metadata)

        # Combine all results into a single string
        final_context = "\n\n".join(mongo_contexts)
        print("\nShortened product list retrieved successfully")

        return final_context

    

    def get_document_retriever(self):
        """
        Retrieve the Chroma vectorstore for the given PDF files, create or update it if necessary.

        Returns:
            document_retriever: The retriever object for querying content from the PDF documents.
        """
        
        # Determine the directory to store and configuration file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        persist_dir = os.path.join(base_dir, "chroma_db")
        config_file = os.path.join(persist_dir, "config.json")

        def load_documents(pdf_paths, chunk_size=500, chunk_overlap=50):
            """
            Read the content of the PDF files and split them into text chunks.

            Args:
                pdf_paths: List of PDF file paths.
                chunk_size: Size of each text chunk.
                chunk_overlap: Overlap between chunks.

            Returns:
                all_chunks: List of text chunks.
            """
            all_chunks = []

            for pdf_path in pdf_paths:
                # Print information about the file being loaded
                print(f"\nLoading {pdf_path}...")
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                
                # Split text into chunks with the given parameters
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len
                )
                chunks = text_splitter.split_documents(documents)
                all_chunks.extend(chunks)

            return all_chunks

        def load_or_create_chroma_vectorstore(pdf_paths, chunk_size=500, chunk_overlap=50):   
            """
            Load the existing Chroma vectorstore or create a new one if the configuration has changed.

            Args:
                pdf_paths: List of PDF file paths.
                chunk_size: Size of each text chunk.
                chunk_overlap: Overlap between chunks.

            Returns:
                vectorstore: The Chroma vectorstore instance.
            """  

            # Check if the persist directory exists
            if os.path.exists(persist_dir):
                print("\nChecking configuration in saved ChromaDB")

                if os.path.exists(config_file):
                    # Read the configuration file
                    with open(config_file, "r") as f:
                        config = json.load(f)
                    
                    # Rebuild if the configuration has changed
                    if config.get("chunk_size") != chunk_size or config.get("chunk_overlap") != chunk_overlap:
                        print("\nConfiguration has changed, updating data!")
                    else:
                        print("\nConfiguration matches, loading existing ChromaDB")
                        vectorstore = Chroma(
                            persist_directory=persist_dir, 
                            embedding_function=self.embeddings, 
                            collection_name="documents"
                        )
                        
                        # Rebuild if the database is empty
                        if vectorstore._collection.count() == 0:
                            print("\nDatabase is empty, needs updating!")
                        else:
                            return vectorstore
                else:
                    print("\nConfiguration file not found, creating new vectorstore!")
            else:
                print("\nPersist directory does not exist, creating new ChromaDB")

            # Load and create chunks from PDF
            chunks = load_documents(
                pdf_paths, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persist_dir,
                collection_name="documents"
            )

            # Create persist directory if it doesn't exist and save configuration
            if not os.path.exists(persist_dir):
                os.makedirs(persist_dir)
            with open(config_file, "w") as f:
                json.dump({
                    "chunk_size": chunk_size, 
                    "chunk_overlap": chunk_overlap
                }, f)
                
            return vectorstore

        # List of PDF files to load
        pdf_dir = os.path.join(base_dir, "document")
        pdf_files = ["Chính sách bảo hành.pdf", "Chính sách đổi trả.pdf"]
        pdf_paths = [os.path.join(pdf_dir, pdf_file) for pdf_file in pdf_files]

        # Initialize retriever from vectorstore with search parameter k = 1
        vectorstore = load_or_create_chroma_vectorstore(pdf_paths, chunk_size=200, chunk_overlap=10)
        document_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

        return document_retriever

    

    def create_qa_chain(self) -> ConversationalRetrievalChain:
        """Create a QA chain with conversational retrieval and custom memory.

        Returns:
            qa_chain: A customized ConversationalRetrievalChain.
        """

        # Custom HumanMessage classes
        class CustomHumanMessage(HumanMessage):
            type: str = "Người dùng"

        # Custom AIMessage classes
        class CustomAIMessage(AIMessage):
            type: str = "Trợ lý ảo"

        class CustomConversationBufferMemory(ConversationBufferMemory):
            """Memory buffer stores conversation history with custom message classes."""

            def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
                """Save the context of the question and answer into memory.
                Args:
                    inputs: A dictionary containing the input, expected to have a key "question".
                    outputs: A dictionary containing the output, expected to have a key "answer".
                """

                # Get the question and answer
                question = inputs.get("question", "") 
                answer = outputs.get("answer", "")     
                if "<|assistant|>" in answer:
                    answer = answer.split("<|assistant|>")[-1].strip()

                # Use custom message classes and add to memory
                human_msg = CustomHumanMessage(content=question)  
                ai_msg = CustomAIMessage(content=answer)         
                self.chat_memory.add_messages([human_msg, ai_msg])  

        # Initialize custom memory
        memory = CustomConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            return_messages=True,
        )
        print("Memory created successfully")

        # Define the system template for the prompt
        system_template = textwrap.dedent(
            """
            <|system|>
            Bạn là Dũng, một trợ lý ảo thông minh của shop thời trang trực tuyến Autumn.
            Nhiệm vụ của bạn là hỗ trợ khách hàng tìm kiếm, tư vấn sản phẩm và giải đáp
            các thắc mắc một cách chính xác và thân thiện.

            **Lịch sử trò chuyện:** {chat_history}

            **Thông tin sản phẩm:**
            {mongo_context}

            **Chính sách:** {document_context}</s>

            <|user|>
            {question}</s>

            <|assistant|>
            """
        ).lstrip()

        # Create prompt combining necessary information
        prompt = PromptTemplate(
            input_variables=[
                "chat_history",
                "question",
                "mongo_context",
                "document_context",
            ],
            template=system_template,
        )

        # Prompt to ensure the model does not alter the question
        no_rephrase_prompt = PromptTemplate(
            template="{question}",
            input_variables=["question"],
        )
        print("Prompt created successfully")

        # Initialize LLM pipeline from HuggingFace
        llm = HuggingFacePipeline(pipeline=self.pipe_tc)
        print("LLM created successfully")

        # Get retriever to query documents
        document_retriever = self.get_document_retriever()

        class IdentityChain(Chain):
            """Chain that takes the question as input and returns the original question."""

            @property
            def input_keys(self) -> List[str]:
                """Input keys of the chain."""
                return ["question"]

            @property
            def output_keys(self) -> List[str]:
                """Output keys of the chain."""
                return ["question"]

            def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
                """Return the question as output."""
                return {"question": inputs["question"]}

        # Create ConversationalRetrievalChain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=document_retriever,
            memory=memory,
            verbose=True,
            condense_question_prompt=no_rephrase_prompt,
            condense_question_llm=None,
            combine_docs_chain_kwargs={
                "prompt": prompt,
                "document_variable_name": "document_context",
            },
        )

        # Replace the question generator with IdentityChain 
        qa_chain.question_generator = IdentityChain()
        print("QA chain created successfully")

        return qa_chain
        

    @staticmethod
    def load_pipeline(model_path: str, gen_config: GenerationConfig) -> Pipeline:
        """
        Returns a text generation pipeline using the TinyLlama model.

        Args:
            model_path: The path to the directory containing the PEFT adapter.
            gen_config: Configuration for text generation parameters.

        Returns:
            text_gen_pipeline: A pipeline for generating text.
        """

        # Load tokenizer and base model
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        tokenizer.pad_token = "<PAD>"
        tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )

        # Load PEFT model into the base_model
        peft_model = PeftModel.from_pretrained(
            base_model,
            model_path,
        )
        peft_model = peft_model.merge_and_unload()

        # Create text generation pipeline with the given parameters
        text_gen_pipeline = pipeline(
            task="text-generation",
            model=peft_model,
            tokenizer=tokenizer,
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            num_return_sequences=gen_config.num_return_sequences,
        )
        print(f"\nSuccessfully created the pipeline with model path: {model_path}")

        return text_gen_pipeline


    
    @staticmethod
    def get_k_filter_conditions(query: str) -> Tuple[int, Optional[Dict[str, Any]]]:
        """
        Determines the number of products and filter conditions from the query string.

        Args:
            query: The query string.

        Returns:
            A tuple consisting of:
                - k: The number of products.
                - filter_conditions: A dictionary of filter conditions using MongoDB syntax.
        """

        # Extract the number of products (k) following the word "danh sách" (list)
        k_match = re.search(r"danh sách\s*(\d+)", query, re.IGNORECASE)
        k = int(k_match.group(1)) if k_match else 1

        # Determine the logical operator
        q_lower = query.lower()
        if "và" in q_lower:
            operator = "$and"
        elif "hoặc" in q_lower:
            operator = "$or"
        else:
            operator = "$and"

        # Regular expression to find filter conditions
        pattern = re.compile(
            r"(?P<field>giá|tồn\s*kho).*?"
            r"(?P<op>bé\s+hơn|lớn\s+hơn)\s+"
            r"(?P<value>\d+)",
            re.IGNORECASE,
        )

        # Iterate through all regex matches in the query
        conditions = []
        for match in pattern.finditer(query):           
            # Normalize input and convert data types
            field_text = match.group("field").strip().lower()
            op_text = match.group("op").strip().lower()
            value = int(match.group("value"))

            # Determine the corresponding database field
            if "giá" in field_text:
                field = "price"
            elif "tồn" in field_text:
                field = "stock"
            else:
                continue

            # Determine the comparison operator
            if "bé hơn" in op_text:
                op = "$lt"
            elif "lớn hơn" in op_text:
                op = "$gt"
            else:
                continue

            conditions.append({field: {op: value}})

        # Package into a dictionary with operator, or set to None if no conditions are found
        filter_conditions = ({operator: conditions} if conditions else None)

        return k, filter_conditions