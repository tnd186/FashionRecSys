import React, { useState, useEffect } from 'react';
import "./FashionWeb.css"; 

const reactionsList = ["👍", "❤️", "😂", "😮", "😢", "😡"]; // Emoji reactions

// Main Component
const FashionWeb = () => {
    // ----- State Management -----
    const [image, setImage] = useState(null); // Uploaded image
    const [isChatBoxVisible, setIsChatBoxVisible] = useState(false); // Show/hide chat
    const [messages, setMessages] = useState([
        { sender: "bot", text: "👋 Chào bạn, mình có thể giúp gì cho bạn ?", reaction: null }
    ]); // Chat messages
    const [hoveredMessageIndex, setHoveredMessageIndex] = useState(null); // Hover state
    const [activeReactionMessage, setActiveReactionMessage] = useState(null); // Message being reacted to
    const [replyingToMessage, setReplyingToMessage] = useState(null); // Message being replied to
    const [isTyping, setIsTyping] = useState(false); // Typing indicator
    const [informationClothes, setInformationClothes] = useState({ result: [] }); // Fashion results
    const [isLoading, setIsLoading] = useState(false); // Loading state
    const [voiceText, setVoiceText] = useState(""); // Voice input
    const [isRecording, setIsRecording] = useState(false); // Mic recording state

    // ----- Handle Image Upload -----
    const handleImageUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            setImage({ file, imageUrl: URL.createObjectURL(file) });
        }
    };

    // ----- Toggle Chat Box Display -----
    const toggleChatBox = () => {
        setIsChatBoxVisible(!isChatBoxVisible);
    };

    // ----- Select Emoji Reaction -----
    const handleReactionSelect = (emoji, messageIndex) => {
        const updatedMessages = messages.map((msg, idx) =>
            idx === messageIndex ? { ...msg, reaction: emoji } : msg
        );
        setMessages(updatedMessages);
        setActiveReactionMessage(null);
    };

    // ----- Reply to Message -----
    const handleReply = (messageIndex) => {
        setReplyingToMessage(messages[messageIndex]);
        const input = document.querySelector("#chat-input input");
        if (input) input.focus();
    };

    // ----- Send Message to Chatbot -----
    const handleSendMessage = async () => {
        const input = document.querySelector("#chat-input input");
        const userMessage = input.value.trim();
    
        if (userMessage) {
            // Add user message to chat
            const newMessage = {
                sender: "user",
                text: userMessage,
                reaction: null,
                replyTo: replyingToMessage ? replyingToMessage.text : null,
            };
            setMessages([...messages, newMessage]);
            input.value = "";
            setReplyingToMessage(null);
    
            // Show bot is typing
            setIsTyping(true);
            setMessages(prev => [...prev, { sender: "bot", text: "...", isTyping: true }]);
    
            try {
                // Send message to backend
                const formData = new FormData();
                formData.append('userMessage', userMessage);
                const chatbotResponse = await fetch('http://127.0.0.1:8000/api/chatbot_response/', {
                    method: 'POST',
                    body: formData,
                });
    
                const botMessage = await chatbotResponse.json();
                const messageChunks = botMessage.result.split('\n').filter(line => line.trim() !== '');
    
                // Format path if it's an image
                const convertToPublicPath = (path) => {
                    const normalizedPath = path.replace(/\\/g, '/');
                    return normalizedPath.replace('D:/Project/FashionRecommendSystem/CrawlData', '');
                };
    
                // Simulate typing delay
                const calculateDelay = (text) => {
                    const words = text.split('').length;
                    return Math.min(6000, words * 40);
                };
    
                setMessages(prev => prev.filter(msg => !msg.isTyping));
                setIsTyping(false);
    
                for (const [index, chunk] of messageChunks.entries()) {
                    const delay = index === 0 ? 0 : calculateDelay(chunk);
                    setMessages(prev => [...prev, { sender: "bot", isTyping: true }]);
                    setIsTyping(true);
    
                    await new Promise(resolve => setTimeout(resolve, delay));
    
                    setMessages(prev => prev.filter(msg => !msg.isTyping));
                    setIsTyping(false);
    
                    const isImagePath = chunk.startsWith("C:/") || chunk.startsWith("D:/") || chunk.startsWith("http");
                    const imagePath = isImagePath ? convertToPublicPath(chunk) : null;
    
                    // Add bot message (text or image)
                    setMessages(prevMessages => [
                        ...prevMessages,
                        isImagePath
                            ? { sender: "bot", image: imagePath, reaction: null }
                            : { sender: "bot", text: chunk, reaction: null },
                    ]);
                }
    
            } catch (error) {
                console.error('Error sending input:', error);
            }
        }
    };
    

    // ----- Voice Recognition (Speech to Text) -----
    const startVoiceRecognition = () => {
        if (!('webkitSpeechRecognition' in window)) {
            alert("Trình duyệt của bạn không hỗ trợ Web Speech API. Vui lòng sử dụng Chrome.");
            return;
        }

        const recognition = new window.webkitSpeechRecognition();
        recognition.lang = "vi-VN";
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        recognition.onstart = () => {
            setIsRecording(true);
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            setVoiceText((prevText) => `${prevText} ${transcript}`.trim());
        };

        recognition.onend = () => {
            setIsRecording(false);
        };

        recognition.start();
    };

    // ----- Auto-scroll chat when new messages appear -----
    useEffect(() => {
        const chatContent = document.getElementById("chat-content");
        if (chatContent) {
            chatContent.scrollTop = chatContent.scrollHeight;
        }
    }, [messages]);

    // ----- Update chat position on scroll -----
    useEffect(() => {
        const frame = document.getElementById("fashion-web-frame");
        const chat = document.getElementById("chat");

        if (!frame || !chat) return;

        const handleScroll = () => {
            let scrollTop = frame.scrollTop;
            chat.style.top = `${scrollTop + window.innerHeight - 108}px`;
        };

        frame.addEventListener("scroll", handleScroll);
        return () => frame.removeEventListener("scroll", handleScroll);
    }, []);

    // ----- Send uploaded image to backend and get fashion results -----
    useEffect(() => {
        const findSimilarityFashion = async () => {
            if (!image) return;

            setIsLoading(true);

            const delayTimeout = setTimeout(async () => {
                try {
                    const formData = new FormData();
                    formData.append('image', image.file);

                    const similarityFashionResponse = await fetch('http://127.0.0.1:8000/api/similarity_fashion_response/', {
                        method: 'POST',
                        body: formData,
                    });

                    const informationClothes = await similarityFashionResponse.json();
                    setInformationClothes(informationClothes);

                } catch (error) {
                    console.error('Error uploading image:', error);
                } finally {
                    setIsLoading(false);
                }
            }, 300);

            return () => clearTimeout(delayTimeout);
        };

        findSimilarityFashion();
    }, [image]);

    // ----- UI Rendering -----
    return (
        <div id="fashion-web-frame">
        {/* Search Bar */}
        <div id="search-bar">
            <button
            id="start-recording"
            className={isRecording ? "pulsing" : ""}
            onClick={startVoiceRecognition}
            />
            <input
            type="text"
            id="voice-input"
            placeholder="Nhập nội dung tìm kiếm hoặc nói..."
            value={voiceText}
            onChange={(e) => setVoiceText(e.target.value)}
            />
        </div>
    
        {/* Upload Image Section */}
        <div
            id="upload-image"
            className={image ? "uploaded" : ""}
            style={{
            background: image
                ? `url(${image.imageUrl}) center/contain no-repeat`
                : "",
            }}
        >
            <input
            type="file"
            accept="image/*"
            id="imageUpload"
            className="upload-input"
            onChange={handleImageUpload}
            />
            {!image && (
            <label htmlFor="imageUpload" className="upload-label">
                <div id="icon-upload" />
                <span id="text-upload">Tải hình lên</span>
            </label>
            )}
        </div>
    
        {/* Output Section */}
        <div id="show-output" className={isLoading ? "loading" : ""}>
            <div className="products-container">
            {informationClothes.result.length > 0 &&
                informationClothes.result.map((item, index) => (
                <div key={index} className="output-item">
                    <img
                    src={`data:image/png;base64,${item.image}`}
                    alt={item.product_name}
                    />
                    <h3>{item.product_name}</h3>
                    <p>Số lượng còn: {item.stock}</p>
                    <p>Giá: {(item.price * 1000).toLocaleString("vi-VN")} đ</p>
                </div>
                ))}
            </div>
            {isLoading && (
            <div className="loading-overlay">
                <div className="spinner" />
            </div>
            )}
        </div>
    
        {/* Chat Button */}
        <div id="chat" onClick={toggleChatBox} />
    
        {/* Chat Box */}
        {isChatBoxVisible && (
            <div id="chat-box">
            {/* Chat Header */}
            <div id="chat-header">
                <span>Shop Thời Trang Autumn</span>
                <button id="close-chat" onClick={toggleChatBox}>×</button>
            </div>
    
            {/* Chat Content */}
            <div id="chat-content">
                {messages.map((msg, index) => (
                <div
                    key={index}
                    className={`chat-message ${msg.sender}`}
                    onMouseEnter={() => setHoveredMessageIndex(index)}
                    onMouseLeave={() => {
                    setHoveredMessageIndex(null);
                    setActiveReactionMessage(null);
                    }}
                >
                    {/* Bot Avatar */}
                    {msg.sender === "bot" && <div className="chat-avatar" />}
    
                    <div className="chat-message-wrapper">
                    {/* Reply Notice */}
                    {msg.replyTo && (
                        <>
                        <div className="reply-notice">Bạn đã phản hồi tin nhắn</div>
                        <div className="reply-bubble">{msg.replyTo}</div>
                        </>
                    )}
    
                    {/* Message Bubble */}
                    <div className="chat-bubble">
                        {msg.isTyping ? (
                        <div className="typing-indicator">
                            <span>.</span><span>.</span><span>.</span>
                        </div>
                        ) : (
                        <>
                            {msg.text && <span>{msg.text}</span>}
                            {msg.image && (
                            <img
                                src={msg.image}
                                alt="Bot response"
                                className="chat-image"
                            />
                            )}
                            {msg.reaction && (
                            <span className="selected-reaction">
                                {msg.reaction}
                            </span>
                            )}
                        </>
                        )}
    
                        {/* Reaction & Reply Icons */}
                        {!msg.isTyping && hoveredMessageIndex === index && (
                        <div className="message-icons">
                            <span
                            className="emoji-icon"
                            onClick={(e) => {
                                e.stopPropagation();
                                setActiveReactionMessage(
                                activeReactionMessage === index ? null : index
                                );
                            }}
                            />
                            <span
                            className="reply-icon"
                            onClick={() => handleReply(index)}
                            />
                        </div>
                        )}
    
                        {/* Reaction Panel */}
                        {!msg.isTyping && activeReactionMessage === index && (
                        <div className="reaction-panel">
                            {reactionsList.map((emoji, i) => (
                            <span
                                key={i}
                                className="reaction-emoji"
                                onClick={() => handleReactionSelect(emoji, index)}
                            >
                                {emoji}
                            </span>
                            ))}
                        </div>
                        )}
                    </div>
                    </div>
                </div>
                ))}
            </div>
    
            {/* Chat Input Area */}
            <div id="chat-input">
                {replyingToMessage && (
                <div className="reply-preview">
                    <span>Đang phản hồi: {replyingToMessage.text}</span>
                    <button onClick={() => setReplyingToMessage(null)}>×</button>
                </div>
                )}
                <div className="input-container">
                <input
                    type="text"
                    placeholder="Nhập tin nhắn..."
                    onKeyDown={(e) => e.key === "Enter" && handleSendMessage()}
                />
                <button id="send-button" onClick={handleSendMessage}>
                    Gửi
                </button>
                </div>
            </div>
            </div>
        )}
    
        {/* Space End Page*/}
        <span id="space-end-page" />
        </div>
    ); 
};

export default FashionWeb;
