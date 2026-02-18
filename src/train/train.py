"""
Training script for baseline Cats vs Dogs model with MLflow tracking.

Inputs:
- data/splits/{train,val,test}.csv
- data/processed images
- params.yaml

Outputs:
- models/model.pt (trained model)
- models/metadata.json (class mapping + config)
- MLflow run logs (params, metrics, artifacts)
"""

import argparse  # CLI arguments
import json  # Save metadata JSON
import time  # Timing
from pathlib import Path  # Paths

import matplotlib.pyplot as plt  # Plotting
import mlflow  # Experiment tracking
import os
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
import numpy as np  # Numerical
import pandas as pd  # Load split CSV
import torch  # PyTorch
import torch.nn as nn  # Loss
import torch.optim as optim  # Optimizer
from sklearn.metrics import (  # Evaluation artifacts
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader, Dataset  # Data pipeline
from torchvision import transforms  # Image transforms
from PIL import Image  # Load images
import yaml  # Read params.yaml

from src.train.model import SimpleCNN  # Import baseline model


class CSVDataset(Dataset):
    """Dataset that reads filepaths+labels from a CSV manifest."""

    def __init__(self, csv_path: Path, transform: transforms.Compose) -> None:
        self.df = pd.read_csv(csv_path)  # Load dataframe from CSV
        self.transform = transform  # Store transform pipeline

    def __len__(self) -> int:
        return len(self.df)  # Number of rows

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]  # Get row
        img_path = row["filepath"]  # Image path string
        label = int(row["label"])  # Label as int

        img = Image.open(img_path).convert("RGB")  # Load image and enforce RGB
        img = self.transform(img)  # Apply transform -> tensor

        return img, label  # Return (tensor, label)


def get_device() -> torch.device:
    """Choose device: prefer MPS if available on Mac, else CPU."""
    if torch.backends.mps.is_available():  # Apple Silicon GPU backend
        return torch.device("mps")  # Use MPS
    return torch.device("cpu")  # Fallback


def plot_and_save_curves(train_losses, val_losses, out_path: Path) -> None:
    """Plot loss curves and save as PNG artifact."""
    plt.figure()  # Create a new figure
    plt.plot(train_losses, label="train_loss")  # Plot train loss
    plt.plot(val_losses, label="val_loss")  # Plot val loss
    plt.xlabel("epoch")  # X axis label
    plt.ylabel("loss")  # Y axis label
    plt.legend()  # Add legend
    plt.tight_layout()  # Better spacing
    plt.savefig(out_path)  # Save to disk
    plt.close()  # Close figure to free memory


def save_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    """Save confusion matrix as a simple plot."""
    plt.figure()  # New figure
    plt.imshow(cm)  # Show matrix as image
    plt.title("Confusion Matrix (Val)")  # Title
    plt.xlabel("Predicted")  # X label
    plt.ylabel("True")  # Y label
    plt.colorbar()  # Color legend
    plt.tight_layout()  # Spacing
    plt.savefig(out_path)  # Save
    plt.close()  # Close


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    """Run one training epoch and return (avg_loss, accuracy)."""
    model.train()  # Set model to train mode
    losses = []  # Store batch losses
    y_true = []  # True labels
    y_pred = []  # Pred labels

    for x, y in loader:  # Iterate batches
        x = x.to(device)  # Move batch to device
        y = y.to(device)  # Move labels to device

        optimizer.zero_grad()  # Clear gradients
        logits = model(x)  # Forward pass
        loss = criterion(logits, y)  # Compute loss
        loss.backward()  # Backprop
        optimizer.step()  # Update parameters

        losses.append(loss.item())  # Track loss

        preds = torch.argmax(logits, dim=1)  # Predicted class indices
        y_true.extend(y.detach().cpu().numpy().tolist())  # Add true labels
        y_pred.extend(preds.detach().cpu().numpy().tolist())  # Add predicted labels

    avg_loss = float(np.mean(losses))  # Average loss
    acc = float(accuracy_score(y_true, y_pred))  # Accuracy
    return avg_loss, acc  # Return metrics


def eval_one_epoch(model, loader, criterion, device) -> tuple[float, float, np.ndarray, str]:
    """Evaluate model and return loss, accuracy, confusion matrix, report."""
    model.eval()  # Eval mode
    losses = []  # Batch losses
    y_true = []  # True labels
    y_pred = []  # Pred labels

    with torch.no_grad():  # No gradients for eval
        for x, y in loader:  # Iterate batches
            x = x.to(device)  # Move to device
            y = y.to(device)  # Move labels

            logits = model(x)  # Forward
            loss = criterion(logits, y)  # Loss

            losses.append(loss.item())  # Track loss
            preds = torch.argmax(logits, dim=1)  # Pred classes

            y_true.extend(y.detach().cpu().numpy().tolist())  # True
            y_pred.extend(preds.detach().cpu().numpy().tolist())  # Pred

    avg_loss = float(np.mean(losses))  # Avg loss
    acc = float(accuracy_score(y_true, y_pred))  # Acc
    cm = confusion_matrix(y_true, y_pred)  # Confusion matrix
    report = classification_report(y_true, y_pred, target_names=["cat", "dog"])  # Report
    return avg_loss, acc, cm, report  # Return everything


def main(params_path: str) -> None:
    """Main training function controlled by params.yaml."""
    params = yaml.safe_load(Path(params_path).read_text())  # Load YAML config

    image_size = int(params["data"]["image_size"])  # Image size
    batch_size = int(params["data"]["batch_size"])  # Batch size
    num_workers = int(params["data"]["num_workers"])  # Workers
    seed = int(params["data"]["seed"])  # Seed

    epochs = int(params["train"]["epochs"])  # Epochs
    lr = float(params["train"]["lr"])  # Learning rate
    weight_decay = float(params["train"]["weight_decay"])  # Weight decay

    exp_name = str(params["mlflow"]["experiment_name"])  # MLflow experiment name

    torch.manual_seed(seed)  # Torch RNG seed
    np.random.seed(seed)  # NumPy RNG seed

    device = get_device()  # Pick device
    print(f"Using device: {device}")  # Print device

    # Define transforms: convert to tensor + normalize
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Ensure correct size
        transforms.ToTensor(),  # Convert PIL -> torch tensor [0,1]
    ])

    train_csv = Path("data/splits/train.csv")  # Train manifest
    val_csv = Path("data/splits/val.csv")  # Val manifest

    train_ds = CSVDataset(train_csv, transform=transform)  # Train dataset
    val_ds = CSVDataset(val_csv, transform=transform)  # Val dataset

    train_loader = DataLoader(  # Train loader
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(  # Val loader
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = SimpleCNN(num_classes=2).to(device)  # Create model
    criterion = nn.CrossEntropyLoss()  # Classification loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Adam optimizer

    # Prepare output dirs
    Path("models").mkdir(parents=True, exist_ok=True)  # Ensure models dir exists
    Path("artifacts").mkdir(parents=True, exist_ok=True)  # Local artifact dir (for plots)

    # Setup MLflow
    mlflow.set_experiment(exp_name)  # Create/select experiment
    with mlflow.start_run():  # Start MLflow run
        # Log parameters to MLflow for reproducibility
        mlflow.log_param("image_size", image_size)  # Log image size
        mlflow.log_param("batch_size", batch_size)  # Log batch size
        mlflow.log_param("epochs", epochs)  # Log epochs
        mlflow.log_param("lr", lr)  # Log LR
        mlflow.log_param("weight_decay", weight_decay)  # Log regularization
        mlflow.log_param("device", str(device))  # Log device used

        train_losses = []  # Store train loss per epoch
        val_losses = []  # Store val loss per epoch

        for epoch in range(epochs):  # Epoch loop
            t0 = time.time()  # Start time

            tr_loss, tr_acc = train_one_epoch(  # Train epoch
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            va_loss, va_acc, cm, report = eval_one_epoch(  # Eval epoch
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            train_losses.append(tr_loss)  # Track train loss
            val_losses.append(va_loss)  # Track val loss

            # Log metrics per epoch
            mlflow.log_metric("train_loss", tr_loss, step=epoch)  # Train loss metric
            mlflow.log_metric("train_acc", tr_acc, step=epoch)  # Train acc metric
            mlflow.log_metric("val_loss", va_loss, step=epoch)  # Val loss metric
            mlflow.log_metric("val_acc", va_acc, step=epoch)  # Val acc metric

            dt = time.time() - t0  # Duration
            print(f"Epoch {epoch+1}/{epochs} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} "
                  f"| va_loss={va_loss:.4f} va_acc={va_acc:.4f} | {dt:.1f}s")  # Print status

        # Save artifacts: loss curves, confusion matrix, report
        curves_path = Path("artifacts/loss_curves.png")  # Curves output
        plot_and_save_curves(train_losses, val_losses, curves_path)  # Save curves
        mlflow.log_artifact(str(curves_path))  # Log artifact

        cm_path = Path("artifacts/confusion_matrix_val.png")  # CM output
        save_confusion_matrix(cm, cm_path)  # Save CM plot
        mlflow.log_artifact(str(cm_path))  # Log artifact

        report_path = Path("artifacts/classification_report_val.txt")  # Report output
        report_path.write_text(report)  # Write report
        mlflow.log_artifact(str(report_path))  # Log artifact

        # Save model (state_dict) as required baseline artifact
        model_path = Path("models/model.pt")  # Model output
        torch.save(model.state_dict(), model_path)  # Save weights
        mlflow.log_artifact(str(model_path))  # Log model artifact too

        # Save metadata for inference reproducibility
        metadata = {  # Metadata dictionary
            "classes": {"0": "cat", "1": "dog"},  # Class mapping
            "image_size": image_size,  # Input size
        }
        metadata_path = Path("models/metadata.json")  # Metadata file
        metadata_path.write_text(json.dumps(metadata, indent=2))  # Write JSON
        mlflow.log_artifact(str(metadata_path))  # Log metadata
