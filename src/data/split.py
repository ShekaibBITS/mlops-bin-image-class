"""
Create deterministic train/val/test splits from data/processed.

Output:
- data/splits/train.csv
- data/splits/val.csv
- data/splits/test.csv

Each CSV includes:
- filepath (relative path)
- label (0=cat, 1=dog)
"""

import argparse  # CLI args
from pathlib import Path  # Path utilities
import pandas as pd  # CSV writing
from sklearn.model_selection import train_test_split  # Splitting utility


def collect_samples(processed_dir: Path) -> pd.DataFrame:
    """Collect all image file paths and labels from processed_dir."""
    cat_dir = processed_dir / "cats"  # Cats folder
    dog_dir = processed_dir / "dogs"  # Dogs folder

    cat_files = sorted([p for p in cat_dir.glob("*.jpg")])  # Cats images
    dog_files = sorted([p for p in dog_dir.glob("*.jpg")])  # Dogs images

    rows: list[dict] = []  # Rows for dataframe

    for p in cat_files:  # Loop cats
        rows.append({"filepath": str(p.as_posix()), "label": 0})  # Label cats as 0
    for p in dog_files:  # Loop dogs
        rows.append({"filepath": str(p.as_posix()), "label": 1})  # Label dogs as 1

    df = pd.DataFrame(rows)  # Create dataframe
    if len(df) == 0:  # No data collected
        raise FileNotFoundError("No processed images found under data/processed/{cats,dogs}.")  # Fail

    return df  # Return samples dataframe


def main(processed_dir: str, out_dir: str, seed: int) -> None:
    """
    Split processed images into train/val/test sets with 80/10/10 ratio.
    We do:
      - train vs temp (80/20)
      - temp -> val/test (50/50) => 10/10
    """
    processed_path = Path(processed_dir)  # Processed data path
    out_path = Path(out_dir)  # Splits output path
    out_path.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    df = collect_samples(processed_path)  # Collect samples

    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=seed, stratify=df["label"]  # 80/20 stratified split
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label"]  # Split 20% into 10/10
    )

    # Save splits as CSV (stable ordering improves reproducibility)
    train_df.sort_values("filepath").to_csv(out_path / "train.csv", index=False)  # Train CSV
    val_df.sort_values("filepath").to_csv(out_path / "val.csv", index=False)  # Val CSV
    test_df.sort_values("filepath").to_csv(out_path / "test.csv", index=False)  # Test CSV

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")  # Summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # CLI parser
    parser.add_argument("--processed_dir", type=str, default="data/processed")  # Input processed data
    parser.add_argument("--out_dir", type=str, default="data/splits")  # Output splits directory
    parser.add_argument("--seed", type=int, default=42)  # RNG seed for reproducibility
    args = parser.parse_args()  # Parse args
    main(processed_dir=args.processed_dir, out_dir=args.out_dir, seed=args.seed)  # Run
