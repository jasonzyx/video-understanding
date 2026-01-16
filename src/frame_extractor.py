"""Frame extraction utility for video analysis."""
import base64
import logging
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """Represents a video frame with metadata."""
    timestamp: float  # seconds
    frame_index: int
    image: np.ndarray  # OpenCV image (BGR)


class FrameExtractor:
    """Extract frames from video at specified intervals."""

    def __init__(self, sample_rate: float = 2.5):
        """
        Args:
            sample_rate: Extract one frame every N seconds
        """
        self.sample_rate = sample_rate

    def extract_frames(self, video_path: Path) -> List[Frame]:
        """Extract frames from video at regular intervals.

        Args:
            video_path: Path to video file

        Returns:
            List of Frame objects
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # Calculate frame indices to extract
        frame_interval = int(fps * self.sample_rate)
        if frame_interval < 1:
            frame_interval = 1

        frames = []
        frame_idx = 0

        logger.info(f"Extracting frames from {video_path}")
        logger.info(f"Video: {duration:.1f}s, {fps:.1f} fps, sampling every {self.sample_rate}s")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                frames.append(Frame(
                    timestamp=timestamp,
                    frame_index=frame_idx,
                    image=frame
                ))

            frame_idx += 1

        cap.release()

        logger.info(f"Extracted {len(frames)} frames")
        return frames

    @staticmethod
    def encode_frame_base64(frame: Frame, max_size: Tuple[int, int] = (512, 512)) -> str:
        """Encode frame as base64 for API transmission.

        Args:
            frame: Frame to encode
            max_size: Resize to max dimensions (for cost optimization)

        Returns:
            Base64 encoded JPEG string
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)

        # Resize to reduce payload size
        h, w = image_rgb.shape[:2]
        if h > max_size[1] or w > max_size[0]:
            scale = min(max_size[0] / w, max_size[1] / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image_rgb = cv2.resize(image_rgb, (new_w, new_h))

        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)

        # Encode as JPEG
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        img_bytes = buffer.getvalue()

        # Encode to base64
        base64_str = base64.b64encode(img_bytes).decode('utf-8')
        return base64_str

    @staticmethod
    def batch_frames(frames: List[Frame], batch_size: int) -> List[List[Frame]]:
        """Batch frames for efficient API calls.

        Args:
            frames: List of frames
            batch_size: Number of frames per batch

        Returns:
            List of frame batches
        """
        batches = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batches.append(batch)
        return batches
