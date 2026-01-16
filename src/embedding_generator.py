"""Generate visual embeddings from video frames using CLIP."""
import logging
from pathlib import Path
from typing import Union, List
import numpy as np
import torch
import cv2
from PIL import Image
import open_clip

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate visual embeddings from video frames using CLIP vision encoder."""

    def __init__(
        self,
        model_name: str = 'ViT-B-32',
        pretrained: str = 'openai',
        device: str = None
    ):
        """
        Initialize CLIP vision encoder for embedding generation.

        Args:
            model_name: CLIP model architecture (default: ViT-B-32)
            pretrained: Pretrained weights to use (default: openai)
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Loading CLIP model: {model_name} ({pretrained}) on {self.device}")

        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Get tokenizer for text encoding
        self.tokenizer = open_clip.get_tokenizer(model_name)

        logger.info("✓ CLIP model loaded successfully")

    def encode_video_clip(self, clip_path: Path, num_frames: int = 4) -> np.ndarray:
        """
        Generate embedding from video clip by averaging frames.

        Args:
            clip_path: Path to video clip
            num_frames: Number of frames to sample (default: 4)

        Returns:
            512-dim L2-normalized embedding vector
        """
        # Extract frames from clip
        frames = self._extract_frames(clip_path, num_frames)

        if not frames:
            logger.warning(f"No frames extracted from {clip_path}")
            return np.zeros(512)

        # Encode each frame
        frame_embeddings = []
        for frame in frames:
            emb = self._encode_frame(frame)
            frame_embeddings.append(emb)

        # Average frame embeddings
        clip_embedding = np.mean(frame_embeddings, axis=0)

        # L2 normalize
        clip_embedding = clip_embedding / np.linalg.norm(clip_embedding)

        return clip_embedding

    def _extract_frames(self, clip_path: Path, num_frames: int) -> List[np.ndarray]:
        """
        Extract evenly-spaced frames from video clip.

        Args:
            clip_path: Path to video clip
            num_frames: Number of frames to extract

        Returns:
            List of frame arrays (BGR format)
        """
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            logger.error(f"Cannot open clip: {clip_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return []

        # Calculate frame indices (evenly spaced)
        if num_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = [
                int(i * (total_frames - 1) / (num_frames - 1))
                for i in range(num_frames)
            ]

        # Extract frames
        frames = []
        current_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_idx in frame_indices:
                frames.append(frame)

            current_idx += 1

        cap.release()
        return frames

    def _encode_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Encode a single frame with CLIP vision encoder.

        Args:
            frame: Frame in BGR format (OpenCV)

        Returns:
            512-dim embedding vector (not normalized)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)

        # Preprocess and encode
        with torch.no_grad():
            image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image)
            embedding = image_features.cpu().numpy().squeeze()

        return embedding

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text query with CLIP text encoder.

        Args:
            text: Text query (e.g., "pick and roll", "fast break")

        Returns:
            512-dim L2-normalized embedding vector
        """
        # Tokenize text
        with torch.no_grad():
            text_tokens = self.tokenizer([text]).to(self.device)

            # Encode text
            text_features = self.model.encode_text(text_tokens)

            # L2 normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Convert to numpy
            embedding = text_features.cpu().numpy().squeeze()

        return embedding

    def encode_batch_clips(
        self,
        clip_paths: List[Path],
        num_frames: int = 4
    ) -> np.ndarray:
        """
        Generate embeddings for multiple clips (batched processing).

        Args:
            clip_paths: List of paths to video clips
            num_frames: Number of frames to sample per clip

        Returns:
            Array of shape (N, 512) with L2-normalized embeddings
        """
        embeddings = []

        for clip_path in clip_paths:
            emb = self.encode_video_clip(clip_path, num_frames)
            embeddings.append(emb)

        return np.array(embeddings)

    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality (512 for ViT-B-32)."""
        return self.model.visual.proj.shape[1] if hasattr(self.model.visual, 'proj') else 512
