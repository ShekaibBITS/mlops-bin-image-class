"""
Download the Cats vs Dogs dataset from Kaggle and extract it into data/raw/.

This script is designed to be reproducible:
- It downloads a fixed Kaggle dataset slug
- It extracts into a deterministic folder
- It can be used as a DVC stage
"""

import argparse  # CLI argument parsing
import os  # Path and environment utilities
import shutil  # File operations
import subprocess  # Run Kaggle CLI command
from pathlib import Path  # Robust path handling


def run_command(cmd: list[str]) -> None:
    """Run a shell command and raise an error if it fails."""
    result = subprocess.run(cmd, check=False)  # Run command without auto-raise
    if result.returncode != 0:  # Non-zero means failure
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")  # Raise with context


def main(dataset_slug: str, out_dir: str) -> None:
    """
    Download and extract dataset to out_dir.
    dataset_slug: Kaggle dataset identifier (e.g. "bhavikjikadara/dog-and-cat-classification-dataset")
    out_dir: output directory for extracted files (e.g. "data/raw")
    """
    out_path = Path(out_dir)  # Convert to Path object
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    tmp_zip_dir = out_path / "_kaggle_tmp"  # Temporary download directory
    tmp_zip_dir.mkdir(parents=True, exist_ok=True)  # Ensure tmp directory exists

    # Download dataset as zip using Kaggle CLI (requires ~/.kaggle/kaggle.json)
    run_command([
        "kaggle", "datasets", "download",
        "-d", dataset_slug,             # Dataset slug
        "-p", str(tmp_zip_dir),         # Download path
        "--force"                       # Re-download to ensure consistency
    ])

    # Find the downloaded zip file (Kaggle usually creates one zip per dataset)
    zip_files = list(tmp_zip_dir.glob("*.zip"))  # List zip files in temp dir
    if len(zip_files) == 0:  # No zip found
        raise FileNotFoundError("No zip file found after Kaggle download.")  # Fail early

    zip_path = zip_files[0]  # Take the first zip found (expected exactly one)

    # Extract zip into data/raw (keeps the extracted dataset structure)
    shutil.unpack_archive(str(zip_path), str(out_path))  # Extract archive to raw dir

    # Clean up temp folder to keep data/raw tidy
    shutil.rmtree(tmp_zip_dir, ignore_errors=True)  # Remove temp directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Create CLI parser
    parser.add_argument("--dataset", type=str, required=True)  # Kaggle dataset slug
    parser.add_argument("--out_dir", type=str, default="data/raw")  # Default output directory
    args = parser.parse_args()  # Parse CLI args
    main(dataset_slug=args.dataset, out_dir=args.out_dir)  # Run main
