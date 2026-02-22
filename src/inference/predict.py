# src/inference/predict.py

"""
Inference utilities for Cats vs Dogs model.

Responsibilities:
- Load trained model weights safely (weights_only=True)
- Load metadata (class mapping, image_size)
- Preprocess input image into model tensor
- Run model forward pass and return probabilities + predicted label
"""

import json  # Read metadata.json
import time  # Latency measurements
from dataclasses import dataclass  # Simple config container
from pathlib import Path  # File paths
from typing import Dict, Tuple  # Type hints

import torch  # PyTorch runtime
import torch.nn.functional as F  # Softmax
from PIL import Image  # Image decoding
from torchvision import transforms  # Image transforms

from src.train.model import SimpleCNN  # Model architecture (same as training)


@dataclass
class ModelBundle:
    """
    Bundle that holds everything needed for inference:
    - model (loaded + eval mode)
    - device
    - transform
    - class mapping
    """
    model: torch.nn.Module
    device: torch.device
    transform: transforms.Compose
    classes: Dict[str, str]
    image_size: int


def get_device() -> torch.device:
    """
    Choose device for inference:
    - Prefer MPS on Apple Silicon if available
    - Else CPU
    """
    if torch.backends.mps.is_available():  # Apple Silicon GPU backend
        return torch.device("mps")  # Use MPS
    return torch.device("cpu")  # Fallback to CPU


def load_metadata(metadata_path: Path) -> Tuple[Dict[str, str], int]:
    """
    Load metadata.json which includes:
    - classes mapping (e.g., {"0":"cat","1":"dog"})
    - image_size (e.g., 224)
    """
    meta = json.loads(metadata_path.read_text())  # Load JSON as dict
    classes = meta["classes"]  # Class mapping
    image_size = int(meta["image_size"])  # Image size
    return classes, image_size  # Return parsed metadata


def build_transform(image_size: int) -> transforms.Compose:
    """
    Build deterministic preprocessing for inference:
    - Resize to training image size
    - Convert to tensor
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Deterministic resize
        transforms.ToTensor(),  # Convert PIL -> torch tensor in [0,1]
    ])


def load_model_bundle(models_dir: str = "models") -> ModelBundle:
    """
    Load model weights + metadata from models_dir.
    Expects:
    - models/model.pt (state_dict)
    - models/metadata.json
    """
    models_path = Path(models_dir)  # Base models directory
    weights_path = models_path / "model.pt"  # Weights file
    metadata_path = models_path / "metadata.json"  # Metadata file

    if not weights_path.exists():  # Ensure weights exist
        raise FileNotFoundError(f"Missing model weights: {weights_path}")  # Fail fast
    if not metadata_path.exists():  # Ensure metadata exists
        raise FileNotFoundError(f"Missing model metadata: {metadata_path}")  # Fail fast

    classes, image_size = load_metadata(metadata_path)  # Read metadata
    device = get_device()  # Pick device
    transform = build_transform(image_size)  # Build preprocessing transform

    model = SimpleCNN(num_classes=2)  # Initialize model architecture

    # Safe load: we saved a state_dict, so weights_only=True is correct and more secure
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)  # Load weights only
    model.load_state_dict(state_dict)  # Apply weights
    model.to(device)  # Move model to device
    model.eval()  # Set evaluation mode

    return ModelBundle(model=model, device=device, transform=transform, classes=classes, image_size=image_size)  # Bundle


def preprocess_image(image: Image.Image, bundle: ModelBundle) -> torch.Tensor:
    """
    Convert input PIL image to a model-ready tensor:
    - Force RGB
    - Apply deterministic transform
    - Add batch dimension
    """
    img = image.convert("RGB")  # Ensure RGB
    x = bundle.transform(img)  # Apply transforms -> tensor [C,H,W]
    x = x.unsqueeze(0)  # Add batch dimension -> [1,C,H,W]
    return x  # Return tensor


def predict(image: Image.Image, bundle: ModelBundle) -> Dict:
    """
    Run model inference and return:
    - predicted label (string)
    - probabilities dict
    - latency_ms
    """
    t0 = time.time()  # Start time

    x = preprocess_image(image, bundle)  # Preprocess input
    x = x.to(bundle.device)  # Move tensor to device

    with torch.no_grad():  # No gradients for inference
        logits = bundle.model(x)  # Forward pass -> logits [1,2]
        probs = F.softmax(logits, dim=1).squeeze(0)  # Softmax -> [2]

    # Convert probabilities to Python floats
    prob_cat = float(probs[0].detach().cpu().item())  # Probability for class 0
    prob_dog = float(probs[1].detach().cpu().item())  # Probability for class 1

    pred_idx = int(torch.argmax(probs).detach().cpu().item())  # Predicted class index
    pred_label = bundle.classes[str(pred_idx)]  # Map to label name

    latency_ms = (time.time() - t0) * 1000.0  # Latency in milliseconds

    return {
        "label": pred_label,  # Predicted label
        "probabilities": {"cat": prob_cat, "dog": prob_dog},  # Probability dict
        "latency_ms": latency_ms,  # Inference latency
        "image_size": bundle.image_size,  # Echo preprocessing size
        "device": str(bundle.device),  # Echo device used
    }
