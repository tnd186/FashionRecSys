# Standard library imports
import base64
import os
import pickle
from io import BytesIO
from typing import Optional, Any, Dict, List, Tuple

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as v2
from PIL import Image
from huggingface_hub import PyTorchModelHubMixin
from rembg import new_session, remove
from safetensors.torch import load_file
from transformers import AutoImageProcessor, AutoModelForObjectDetection, SwinConfig, SwinModel

# Django imports
from django.db.models import Q

# ORM Models
from fashion_web_app.models import Product


class SimilarityFashion: 
    def __init__(self, model_path_cd: Optional[str] = None, model_path_cfe: Optional[str] = None,) -> None:
        """
        Initialize the SimilarityFashion object.

        Args:
            model_path_cd: Path to the pretrained folder of the Clothes Detection model.
            model_path_cfe: Path to the pretrained folder of the Clothes Feature Extractor model.
        """

        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set the path to the models
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path_cd is None:
            model_path_cd = os.path.join(base_dir, "weights", "Clothes_Detection")
        if model_path_cfe is None:
            model_path_cfe = os.path.join(base_dir, "weights", "Clothes_Feature_Extractor")

        # Initialize the clothes detection model
        self.image_processor_cd = AutoImageProcessor.from_pretrained(model_path_cd)
        self.model_cd = AutoModelForObjectDetection.from_pretrained(model_path_cd)

        # Initialize the configuration and processor for the feature extractor
        self.config_cfe = SwinConfig.from_pretrained(model_path_cfe)
        self.image_processor_cfe = AutoImageProcessor.from_pretrained(model_path_cfe)

        # Define the ImageEncoder class
        class ImageEncoder(nn.Module, PyTorchModelHubMixin):
            """ImageEncoder generates image embeddings using the Swin model"""

            def __init__(self, config: SwinConfig,) -> None:
                """
                Initializes the model with a Swin backbone and an embedding layer.

                Args:
                    config: Configuration object for the Swin transformer backbone.
                """
                super(ImageEncoder, self).__init__()
                self.swin = SwinModel(config=config)
                self.embedding_layer = nn.Linear(config.hidden_size, 128)

            def forward(self, image_tensor: torch.Tensor,) -> torch.Tensor:
                """
                Performs a forward pass and returns the normalized embeddings.

                Args:
                    image_tensor: Input image tensor of shape (batch_size, channels, height, width).

                Returns:
                    torch.Tensor: L2-normalized embedding of shape (batch_size, 128).
                """                
                features = self.swin(image_tensor).pooler_output
                embeddings = self.embedding_layer(features)
                return F.normalize(embeddings, p=2, dim=1)

        # Load the feature extractor
        self.model_cfe = ImageEncoder(self.config_cfe)
        weights_file = os.path.join(model_path_cfe, "model.safetensors")
        state_dict = load_file(weights_file)
        self.model_cfe.load_state_dict(state_dict)

        # Set up image transforms
        self.transform = v2.Compose(
            [
                v2.Resize(
                    (
                        self.config_cfe.image_size,
                        self.config_cfe.image_size,
                    )
                ),
                v2.ToTensor(),
                v2.Normalize(
                    mean=self.image_processor_cfe.image_mean,
                    std=self.image_processor_cfe.image_std,
                ),
            ]
        )

        # Load embeddings and paths
        emb_path = os.path.join(model_path_cfe, "category_image_embeddings.pkl")
        paths_path = os.path.join(model_path_cfe, "category_image_paths.pkl")
        with open(emb_path, "rb") as f:
            self.category_image_embeddings = pickle.load(f)
        with open(paths_path, "rb") as f:
            self.category_image_paths = pickle.load(f)

        # Create session for background removal
        self.session = new_session(model_name="isnet-general-use")


    def __call__(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Based on the input image, identify related clothing products and return the images and information of those products.

        Args:
            image: Input image.

        Returns:
            results: A list of results, each dict includes: base64-encoded image and product information.
        """

        # Identify image regions containing clothing by category
        results = []
        detected_clothes = self.detect_objects(image)

        # Group cropped images by predefined labels
        labels = ["hat", "outer", "top", "bottom", "shoes", "bag"]
        cropped_images_by_label = {label: [] for label in labels}
        for cropped_image, label in detected_clothes:
            label_lower = label.lower()
            if label_lower in cropped_images_by_label:
                cropped_images_by_label[label_lower].append(cropped_image)

        # Iterate over each label group and process
        for label, cropped_images in cropped_images_by_label.items():
            if not cropped_images:
                continue

            print(f"Processing category: {label} with {len(cropped_images)} images")

            # Convert cropped images to tensors for model input
            image_tensors = [self.transform(img) for img in cropped_images]
            cropped_image_tensors = torch.stack(image_tensors)

            # Compute embeddings without gradients to save memory
            with torch.no_grad():
                embeddings = self.model_cfe(cropped_image_tensors).tolist()
            embeddings_array = np.array(embeddings)

            # Find nearest indices in the category's embeddings
            category_key = label.capitalize()
            category_embeddings = self.category_image_embeddings[category_key]
            nearest_indices = self.find_nearest_neighbors(
                embeddings_array,
                category_embeddings
            )

            # Build result list, avoiding duplicate products
            for idx in nearest_indices[:, 0]:
                img_path = self.category_image_paths[category_key][idx]

                # Skip if result with this img_path already exists
                if any(item["img_path"] == img_path for item in results):
                    continue

                # Remove background and collect product info
                output_image = self.remove_background(img_path, label)
                product_info = self.get_information_product(output_image, img_path)
                results.append(product_info)

        return results


    def detect_objects(self, img: Image.Image) -> List[Tuple[Image.Image, str, float]]:
        """Detects objects in the image and returns cropped sub-images with labels and confidence scores.

        Args:
            img: The input image in which to detect objects.

        Returns:
            A list of tuples containing:
                - Cropped sub-image (PIL.Image.Image)
                - Object label (str)
                - Confidence score (float)
        """

        # Convert the input image to tensor and call the model for prediction
        with torch.no_grad():
            tensor_input = self.image_processor_cd(
                images=[img],
                return_tensors="pt"
            )
            output = self.model_cd(**tensor_input)

            # Target size for post-processing
            dims = torch.tensor([[img.height, img.width]])
            processed = self.image_processor_cd.post_process_object_detection(
                output,
                threshold=0.35,
                target_sizes=dims
            )[0]

        # Convert results to a list of (score, label_id, box)
        raw_detections = [
            (s.item(), l.item(), [b.item() for b in box])
            for s, l, box in zip(
                processed["scores"],
                processed["labels"],
                processed["boxes"]
            )
        ]

        # Apply non-max suppression to remove duplicates
        nms_detections = self.non_max_suppression(raw_detections)

        # Define default padding ratios for each label
        padding_ratios = {
            "bag": 0.1,
            "hat": 0.1,
            "bottom": 0.1,
            "outer": 0.1,
            "shoes": -0.05
        }

        # If label 'outer' exists, reduce top padding; otherwise increase top padding
        has_outer = any(
            self.model_cd.config.id2label[label_id] == "outer"
            for _, label_id, _ in nms_detections
        )
        padding_ratios["top"] = -0.1 if has_outer else 0.1

        # Iterate through each detection filtered by non-max suppression
        cropped_images = []
        for score, label_id, box in nms_detections:
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min

            # Get label name and corresponding padding ratio
            label_str = self.model_cd.config.id2label[label_id]
            pad_ratio = padding_ratios.get(label_str, 0.0)

            # Calculate new coordinates with padding
            x_min = max(0, x_min - width * pad_ratio)
            y_min = max(0, y_min - height * pad_ratio)
            x_max = min(img.width, x_max + width * pad_ratio)
            y_max = min(img.height, y_max + height * pad_ratio)

            # Crop the image and add to result list
            cropped_img = img.crop((x_min, y_min, x_max, y_max))
            cropped_images.append((cropped_img, label_str, score))

        return cropped_images


    def remove_background(self, image_path: str, label: str) -> Image.Image:
        """
        Removes the background from an image and returns a centered image with a transparent background.

        Args:
            image_path: Path to the input image file.
            label: Label indicating the product category.

        Returns:
            centered_img: The image with background removed and centered on a transparent canvas.
        """
        # Open the image and convert it to RGBA format
        img = Image.open(image_path).convert("RGBA")

        # Set parameters for background removal
        remove_kwargs = {
            "alpha_matting": True,
            "alpha_matting_erode_size": 2,
            "post_process_mask": True,
        }

        if label not in ("top", "bottom", "outer", "shoes"):
            remove_kwargs["session"] = self.session

        img_no_bg = remove(img, **remove_kwargs)

        # Crop to the bounding box of the remaining foreground
        bbox = img_no_bg.getbbox()
        if bbox:
            img_no_bg = img_no_bg.crop(bbox)

        # Create a square canvas and center the image on it
        max_side = max(img_no_bg.size)
        canvas_size = int(max_side * 1.2)
        centered_img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

        offset = (
            (canvas_size - img_no_bg.width) // 2,
            (canvas_size - img_no_bg.height) // 2,
        )
        centered_img.paste(img_no_bg, offset, mask=img_no_bg)

        return centered_img


    @staticmethod
    def calc_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Args:
            box_a: Format (x_min, y_min, x_max, y_max) of box A.
            box_b: Format (x_min, y_min, x_max, y_max) of box B.

        Returns:
            float: The IoU value between the two boxes, in the range [0.0, 1.0].
        """

        # Calculate the area of each box
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        # Calculate the intersection area
        x_overlap = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
        y_overlap = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
        intersection = x_overlap * y_overlap

        # Calculate the union area
        union = area_a + area_b - intersection
        if union <= 0.0:
            return 0.0  # avoid division by zero

        return intersection / union


    @staticmethod
    def non_max_suppression(
        boxes: List[Tuple[float, Tuple[float, float, float, float]]],
        iou_threshold: float = 0.7
    ) -> List[Tuple[float, Tuple[float, float, float, float]]]:
        """
        Perform Non-Maximum Suppression (NMS) on a list of scored bounding boxes.

        Args:
            boxes: List of tuples (score, (x_min, y_min, x_max, y_max)).
            iou_threshold: IoU threshold for suppressing overlapping boxes.

        Returns:
            keep: List of boxes retained after applying NMS.
        """

        # Sort boxes by score in descending order
        sorted_boxes = sorted(boxes, key=lambda item: item[0], reverse=True)
        keep = []
        removed = [False] * len(sorted_boxes)

        # Iterate through sorted boxes
        for i, (score_i, box_i) in enumerate(sorted_boxes):
            if removed[i]:
                continue
            # If the box is not removed, keep it
            keep.append((score_i, box_i))
            # Suppress boxes with high IoU
            for j in range(i + 1, len(sorted_boxes)):
                if removed[j]:
                    continue
                _, box_j = sorted_boxes[j]
                if SimilarityFashion.calc_iou(box_i, box_j) >= iou_threshold:
                    removed[j] = True  # mark as removed

        return keep


    @staticmethod
    def get_information_product(output_image: Image.Image, img_path: str) -> Dict[str, Any]:
        """
        Convert an image to base64 and query product information.

        Args:
            output_image: The image in Image.Image format.
            img_path: Path to the image.

        Returns:
            Product information including:
                - image (str): Image in base64 format.
                - product_name (str): Name of the product.
                - img_path (str): Image path.
                - stock (int): Available stock quantity.
                - price (float): Product price.
        """
        # Encode image to base64
        buffer_ = BytesIO()
        output_image.save(buffer_, format="PNG")
        img_str = base64.b64encode(buffer_.getvalue()).decode()

        # Normalize path and get the file name
        normalized_path = img_path.replace("\\", "/")
        filename = os.path.basename(normalized_path)

        # Query Product object by filename
        product = Product.objects.get(
            Q(positive_url__icontains=filename)
        )

        return {
            "image": img_str,
            "product_name": product.product_name,
            "img_path": img_path,
            "stock": product.stock,
            "price": product.price,
        }


    @staticmethod
    def find_nearest_neighbors(query_embeddings: np.ndarray, target_embeddings: np.ndarray, top_k: int = 1) -> np.ndarray:
        """
        Compute the Euclidean distance between query embeddings and target embeddings,
        and return the indices of the top_k nearest neighbors.

        Args:
            query_embeddings: Embedding array of the query, shape (n_queries, dim).
            target_embeddings: Embedding array of the targets, shape (n_targets, dim).
            top_k: Number of nearest neighbors to return.

        Returns:
            nearest_indices: Array of indices of the nearest neighbors with shape (n_queries, top_k).
        """
        # Compute the Euclidean distance matrix from each query to each target
        distances = np.linalg.norm(
            query_embeddings[:, np.newaxis] - target_embeddings,
            axis=2
        )

        # Sort to get the smallest (nearest) indices
        nearest_indices = np.argsort(distances, axis=1)[:, :top_k]

        return nearest_indices