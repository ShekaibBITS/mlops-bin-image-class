"""
Preprocess raw images into standardized 224x224 RGB images.

Input:  data/raw (downloaded from Kaggle)
Output: data/processed/cats and data/processed/dogs

We keep filenames deterministic to support reproducibility.
"""

import argparse  # CLI parsing
from pathlib import Path  # Robust file paths
from PIL import Image  # Image loading & resizing
from tqdm import tqdm  # Progress bar


def is_image_file(path: Path) -> bool:
    """Return True if file extension looks like an image."""
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"}  # Supported image extensions


def find_images(root: Path) -> list[Path]:
    """Recursively find image files under root."""
    return [p for p in root.rglob("*") if p.is_file() and is_image_file(p)]  # Collect images


def infer_label_from_path(path: Path) -> str | None:
    """
    Infer label from file path using common dataset conventions.
    Returns "cats", "dogs", or None if unknown.
    """
    s = str(path).lower()  # Lowercase string version of path
    if "cat" in s and "dog" not in s:  # Path contains cat but not dog
        return "cats"  # Label as cats
    if "dog" in s and "cat" not in s:  # Path contains dog but not cat
        return "dogs"  # Label as dogs
    # If ambiguous (both words or neither), return None and skip
    return None


def preprocess_one(in_path: Path, out_path: Path, size: int) -> None:
    """Load an image, convert to RGB, resize, and save as JPEG."""
    img = Image.open(in_path)  # Open image from disk
    img = img.convert("RGB")  # Force 3-channel RGB
    img = img.resize((size, size))  # Resize to (224,224) or other size
    out_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure output folder exists
    img.save(out_path, format="JPEG", quality=95)  # Save standardized JPEG


def main(raw_dir: str, processed_dir: str, size: int) -> None:
    """
    Preprocess all valid images found in raw_dir.
    Writes output into processed_dir/cats and processed_dir/dogs.
    """
    raw_path = Path(raw_dir)  # Raw dataset directory
    out_root = Path(processed_dir)  # Output directory

    images = find_images(raw_path)  # Find all image files
    if len(images) == 0:  # Fail if no images found
        raise FileNotFoundError(f"No images found under {raw_dir}")  # Raise helpful error

    kept = 0  # Count kept images
    skipped = 0  # Count skipped images

    for in_path in tqdm(images, desc="Preprocessing images"):  # Progress loop
        label = infer_label_from_path(in_path)  # Try to infer label from filename/path
        if label is None:  # Unknown/ambiguous
            skipped += 1  # Increment skipped
            continue  # Skip this file

        # Create a deterministic output filename:
        # - preserve original stem
        # - add parent folder stem to reduce collisions
        out_name = f"{in_path.parent.stem}_{in_path.stem}.jpg"  # Deterministic name
        out_path = out_root / label / out_name  # Output path by label

        try:
            preprocess_one(in_path=in_path, out_path=out_path, size=size)  # Transform + save
            kept += 1  # Increment kept
        except Exception:
            skipped += 1  # If corrupted/unreadable, skip safely

    if kept == 0:  # If nothing processed, fail
        raise RuntimeError("No images were successfully preprocessed. Check dataset structure.")  # Error

    print(f"Processed: {kept} images | Skipped: {skipped} images")  # Summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # CLI parser
    parser.add_argument("--raw_dir", type=str, default="data/raw")  # Input raw directory
    parser.add_argument("--processed_dir", type=str, default="data/processed")  # Output directory
    parser.add_argument("--size", type=int, default=224)  # Resize dimension
    args = parser.parse_args()  # Parse args
    main(raw_dir=args.raw_dir, processed_dir=args.processed_dir, size=args.size)  # Run
