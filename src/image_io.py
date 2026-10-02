"""The shared JPEG decoding contract for training and inference."""

import numpy as np
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def decode_semantic_image(source):
    """Decode a path or binary file into normalized float32 RGB CHW pixels."""
    with Image.open(source) as image:
        if image.format != "JPEG" or image.mode != "RGB" or image.size != (224, 224):
            raise ValueError("Expected a 224x224 RGB JPEG cache image")
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    return np.ascontiguousarray((pixels - IMAGENET_MEAN) / IMAGENET_STD)
