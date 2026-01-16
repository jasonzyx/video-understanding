"""Generate trajectory embeddings from motion descriptions using CLIP."""
import logging
from typing import Union
import numpy as np
import torch
import open_clip

logger = logging.getLogger(__name__)


class TrajectoryEmbeddingGenerator:
    """Generate trajectory embeddings from motion descriptions using CLIP text encoder."""

    def __init__(
        self,
        model_name: str = 'ViT-B-32',
        pretrained: str = 'openai',
        device: str = None
    ):
        """
        Initialize CLIP text encoder for trajectory embedding generation.

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

        # Get tokenizer
        self.tokenizer = open_clip.get_tokenizer(model_name)

        logger.info("✓ CLIP text encoder loaded successfully")

    def generate_trajectory_embedding(
        self,
        movement_description: str
    ) -> np.ndarray:
        """
        Generate trajectory embedding from GPT-4V motion description.

        The trajectory embedding captures motion semantics:
        - "player cuts to basket" vs "screen set at high post" vs "standing still"
        - "fast break" vs "half court offense"
        - "drive penetration" vs "perimeter shooting"

        Args:
            movement_description: Text description of player movements
                                 (e.g., "player cuts to basket, screen set at high post")

        Returns:
            512-dim L2-normalized embedding vector
        """
        # Tokenize text
        with torch.no_grad():
            text_tokens = self.tokenizer([movement_description]).to(self.device)

            # Encode text with CLIP
            text_features = self.model.encode_text(text_tokens)

            # L2 normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Convert to numpy
            embedding = text_features.cpu().numpy().squeeze()

        return embedding

    def generate_batch_embeddings(
        self,
        movement_descriptions: list[str]
    ) -> np.ndarray:
        """
        Generate trajectory embeddings for multiple descriptions (batched).

        Args:
            movement_descriptions: List of text descriptions

        Returns:
            Array of shape (N, 512) with L2-normalized embeddings
        """
        if not movement_descriptions:
            return np.array([])

        # Tokenize all texts
        with torch.no_grad():
            text_tokens = self.tokenizer(movement_descriptions).to(self.device)

            # Encode texts with CLIP
            text_features = self.model.encode_text(text_tokens)

            # L2 normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Convert to numpy
            embeddings = text_features.cpu().numpy()

        return embeddings

    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality (512 for ViT-B-32)."""
        return self.model.text.text_projection.shape[1]
