import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import math



class PalletMultiViewDataset(Dataset):
    """
    A PyTorch Dataset for loading multi-view image data of pallets.
    
    This dataset expects a directory structure where each subdirectory 
    represents a unique 'scene' containing multiple camera views.
    
    Attributes:
        root_dir (str): Path to the dataset root directory.
        transform (callable, optional): Transformations to be applied to images.
        scenes (list): Sorted list of scene folder names.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Identify subdirectories and apply natural/numerical sorting
        # Handles naming conventions like "Set1", "Set2", etc.
        self.scenes = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))],
            key=lambda x: int(x.replace("Set", "")) if x.replace("Set", "").isdigit() else x
        )

    def __len__(self):
        """Returns the total number of scenes in the dataset."""
        return len(self.scenes)

    def __getitem__(self, idx):
        """
        Retrieves a single scene's multi-view images and its corresponding label.
        
        Args:
            idx (int): Index of the scene to retrieve.
            
        Returns:
            tuple: (stacked_images, label_tensor, scene_name)
        """
        scene_name = self.scenes[idx]
        folder = os.path.join(self.root_dir, scene_name)

        # Filter and sort image files within the scene folder
        img_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])

        # Validation: Ensure the scene contains the required number of views
        if len(img_files) < NUM_VIEWS:
            raise ValueError(f"Scene '{scene_name}' contains only {len(img_files)} images, "
                             f"but {NUM_VIEWS} are required.")

        # Label Extraction: Parses the pallet count from the filename metadata
        # Expects format: prefix_id_LABEL.extension
        label = float(img_files[0].split("_")[2].split(".")[0])

        images = []
        for i in range(NUM_VIEWS):
            img_path = os.path.join(folder, img_files[i])
            img = Image.open(img_path).convert("RGB")
            
            if self.transform:
                img = self.transform(img)
            
            images.append(img)

        # Return a 4D tensor (Views x Channels x Height x Width) and the scalar label
        return (
            torch.stack(images), 
            torch.tensor([label], dtype=torch.float32), 
            scene_name
        )


import torch
import torch.nn as nn

class MultiViewAttention(nn.Module):
    """
    Transformer-based attention mechanism for aggregating multi-view features.
    
    This module applies a self-attention mechanism across the view dimension (V)
    to capture spatial and semantic relationships between different camera angles
    within a single scene.
    
    Args:
        embed_dim (int): The dimensionality of the input features (D). Default: 384.
        num_heads (int): Number of attention heads in the Multi-Head Attention. Default: 4.
        num_layers (int): Number of Transformer Encoder layers to stack. Default: 2.
    """

    def __init__(self, embed_dim: int = 384, num_heads: int = 4, num_layers: int = 2):
        super().__init__()

        # Define the configuration for a single encoder layer
        # Using GELU activation for smoother gradients compared to ReLU
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu"
        )

        # Stack the encoder layers to create the full attention module
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-view feature aggregation.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, V, D]
                              B: Batch Size
                              V: Number of Views
                              D: Embedding Dimension (Feature size)
        
        Returns:
            torch.Tensor: Aggregated feature tensor of shape [B, V, D] 
                          where each view's feature is refined by context 
                          from all other views.
        """
        # The Transformer Encoder processes the view dimension via self-attention
        return self.encoder(x)

import torch
import torch.nn as nn

class DinoRegressorPRO(nn.Module):
    """
    Multi-view Regression Model utilizing DINOv2 features and Transformer fusion.
    
    This architecture extracts high-dimensional spatial features from multiple
    views using a frozen DINOv2 backbone, fuses them via self-attention, 
    and predicts a continuous scalar value (e.g., pallet count).
    """

    def __init__(self):
        super().__init__()

        # 1. Feature Extraction (Backbone)
        # Loading pre-trained DINOv2 (Vision Transformer Small - ViT-S/14)
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        
        # Freeze backbone parameters to leverage pre-trained weights without updating them
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.embed_dim = self.backbone.embed_dim  # Standard ViT-S dim is 384

        # 2. Multi-View Transformer Fusion
        # Contextualizes features across views using the previously defined Attention module
        self.fusion = MultiViewAttention(
            embed_dim=self.embed_dim, 
            num_heads=4, 
            num_layers=2
        )

        # 3. Regression Head
        # Maps the fused global representation to a final scalar output
        self.regressor = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, V, C, H, W]
               (Batch, Views, Channels, Height, Width)
        
        Returns:
            torch.Tensor: Scalar prediction for each sample [B, 1]
        """
        b, v, c, h, w = x.shape

        # Flatten Batch and Views for backbone processing
        x = x.reshape(b * v, c, h, w)

        # Extract [CLS] token features: [B*V, 384]
        feats = self.backbone.forward_features(x)["x_norm_clstoken"]
        
        # Reshape back to separate multi-view dimensions: [B, V, 384]
        feats = feats.reshape(b, v, self.embed_dim)

        # Apply Transformer-based fusion across the view dimension
        fused = self.fusion(feats)  # [B, V, 384]

        # Global average pooling across views to get a single scene representation
        pooled = fused.mean(dim=1)  # [B, 384]

        # Predict final value
        return self.regressor(pooled)

import matplotlib.pyplot as plt
import torch

def normalize_targets(labels: torch.Tensor):
    """
    Standardizes the target labels using Z-score normalization.
    
    Normalizing targets helps the model converge faster by ensuring 
    the loss function operates on a consistent scale.
    
    Args:
        labels (torch.Tensor): Raw target values.
        
    Returns:
        tuple: (normalized_labels, mean, std)
    """
    mean = labels.mean()
    std = labels.std()
    # Z-score formula: (x - μ) / σ
    normalized_labels = (labels - mean) / std
    return normalized_labels, mean, std

def plot_metrics(train_losses: list, val_maes: list):
    """
    Generates diagnostic plots for training loss and validation accuracy.
    
    Provides a side-by-side comparison to detect overfitting or 
    underfitting during the training process.
    
    Args:
        train_losses (list): Training loss recorded at each epoch.
        val_maes (list): Validation Mean Absolute Error recorded at each epoch.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot 1: Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker='o', linestyle='-', label='Training Loss')
    plt.title("Training Loss Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Plot 2: Validation MAE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_maes, marker='o', color='orange', linestyle='-', label='Val MAE')
    plt.title("Validation Mean Absolute Error")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (Standardized Units)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

def evaluate(model: nn.Module, loader: DataLoader, mean: float, std: float):
    """
    Performs a comprehensive evaluation of the model on a given dataset.
    
    Metrics calculated:
    - MAE: Mean Absolute Error (Directly interpretable as pallet count error).
    - RMSE: Root Mean Squared Error (Penalizes larger outliers).
    - MARE: Mean Absolute Relative Error (Error percentage relative to target).
    """
    model.eval()
    preds_list = []
    trues_list = []

    with torch.no_grad():
        for imgs, labels, names in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Inference on normalized space
            pred_norm = model(imgs)
            
            # Denormalization: Convert predictions back to actual pallet counts
            pred = pred_norm * std + mean
            
            preds_list.append(pred.item())
            trues_list.append(labels.item())

    preds = torch.tensor(preds_list)
    trues = torch.tensor(trues_list)

    # Statistical Metric Calculations
    mae = torch.mean(torch.abs(preds - trues))
    rmse = torch.sqrt(torch.mean((preds - trues) ** 2))
    mare = torch.mean(torch.abs(trues - preds) / (trues + 1e-7))

    print(f"\n{'='*5} FULL EVALUATION {'='*5}")
    print(f"MAE:   {mae:.2f}")
    print(f"RMSE:  {rmse:.2f}")
    print(f"MARE:  {mare:.4f} (~{mare*100:.2f}%)")

    print("\nSample Predictions vs Ground Truth:")
    for p, t in zip(preds_list[:10], trues_list[:10]):  # Showing first 10 for brevity
        print(f"Pred: {p:.2f} | True: {t:.2f} | Diff: {abs(p - t):.2f}")

def run_project():
    """
    Main orchestration function: Data prep, Model init, Training, and Evaluation.
    """
    # 1. Pre-processing Pipeline
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # 2. Data Preparation
    dataset = PalletMultiViewDataset("./DataSet", transform=transform)
    # 80/10/10 Split
    train_ds, val_ds, test_ds = random_split(dataset, [8, 1, 1])

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1)
    test_loader  = DataLoader(test_ds, batch_size=1)

    # Calculate global target statistics for Z-score normalization
    all_labels = torch.tensor([dataset[i][1].item() for i in range(len(dataset))])
    _, mean, std = normalize_targets(all_labels)

    # 3. Model & Optimization Setup
    model = DinoRegressorPRO().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.L1Loss()  # Robust against outliers compared to MSE

    print(f"Starting Training... GPU Available: {torch.cuda.is_available()}")

    train_losses = []
    val_maes = []

    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for imgs, labels, train_name in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            # Normalize target for this batch
            labels_norm = (labels - mean) / std

            optimizer.zero_grad()
            pred_norm = model(imgs)
            loss = criterion(pred_norm, labels_norm)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_losses.append(total_loss)

        # 5. Periodic Validation (Every 10 epochs)
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_mae_epoch = 0
            with torch.no_grad():
                for imgs, labels, val_name in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    
                    # Get prediction and denormalize for human-readable MAE
                    pred = model(imgs) * std + mean
                    val_mae_epoch += abs(pred.item() - labels.item())
            
            val_maes.append(val_mae_epoch / len(val_loader))
        else:
            val_maes.append(None)

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    # 6. Final Test and Visualization
    evaluate(model, test_loader, mean, std)
    plot_metrics(train_losses, val_maes)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_metrics(train_losses: list, val_maes: list):
    """
    Generate production-quality diagnostic plots for model training and validation.
    
    Features:
    - Adaptive smoothing (SMA) for loss trends.
    - Automated 'Best Model' annotation for validation metrics.
    - Professional Seaborn aesthetics for technical reporting.
    """
    # Set sophisticated visual style for professional reporting
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["font.family"] = "sans-serif"
    
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Handle sparse validation data (aligning non-None values with their respective epochs)
    val_epochs = [i + 1 for i, v in enumerate(val_maes) if v is not None]
    val_values = [v for v in val_maes if v is not None]

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- SUBPLOT 1: Training Loss Convergence ---
    # Plot raw loss with low alpha to show variance
    sns.lineplot(x=epochs, y=train_losses, ax=ax[0], color='#2c3e50', alpha=0.3, label='Raw Loss')
    
    # Apply Simple Moving Average (SMA) to highlight the learning trajectory
    if len(train_losses) > 5:
        smoothed_loss = pd.Series(train_losses).rolling(window=5, min_periods=1).mean()
        sns.lineplot(x=epochs, y=smoothed_loss, ax=ax[0], color='#e74c3c', linewidth=2.5, label='Trend (SMA-5)')
    
    ax[0].set_title("Training Loss Convergence", pad=20, fontweight='bold')
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss Magnitude")
    ax[0].legend(frameon=True, loc='upper right')

    # --- SUBPLOT 2: Validation Performance (MAE) ---
    if val_values:
        # Plot validation points connected by a high-contrast line
        sns.lineplot(x=val_epochs, y=val_values, ax=ax[1], color='#2980b9', 
                     marker='o', markersize=8, linewidth=2.5, label='Validation MAE')
        
        # Identify and annotate the optimal performance point
        best_mae = min(val_values)
        best_epoch = val_epochs[val_values.index(best_mae)]
        
        # Add dynamic annotation for the 'Best Model' checkpoint
        ax[1].annotate(f'Best MAE: {best_mae:.2f}', 
                       xy=(best_epoch, best_mae), 
                       xytext=(best_epoch, best_mae + (max(val_values) - min(val_values)) * 0.15),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=1.2, headwidth=8),
                       horizontalalignment='center', 
                       fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2980b9", alpha=0.8))

    ax[1].set_title("Validation Accuracy (MAE)", pad=20, fontweight='bold')
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Mean Absolute Error")
    
    # Clean up plot borders for a modern look
    sns.despine()
    plt.tight_layout()
    
    # Optional: Export at high resolution for presentations
    # plt.savefig("model_performance_report.png", dpi=300, bbox_inches='tight')
    plt.show()


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import copy

def run_project():
    """
    Main orchestration function for model training, validation, and convergence.
    
    This pipeline implements:
    1. Data Preprocessing & Partitioning (80/10/10 ratio).
    2. Dynamic Target Normalization (Z-score scaling).
    3. Training loop with Early Stopping based on Validation MAE.
    4. Best Model Checkpointing (Automatic restoration of optimal weights).
    """
    
    # --- Configuration & Hyperparameters ---
    MAX_EPOCHS = 10000    # High limit to allow full convergence to global minimum
    PATIENCE = 50         # Stop if no improvement after 50 validation cycles
    VALIDATION_FREQ = 5   # Frequency of validation performance checks
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data Transformation Pipeline
    # Using 224x224 as the standard input resolution for DINOv2
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
    ])

    # 2. Dataset Initialization & Splitting
    dataset = PalletMultiViewDataset("./DataSet", transform=transform)
    
    # Splitting logic to ensure a dedicated test set for final evaluation
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1)
    
    # 3. Target Statistics for Z-score Normalization
    # Normalizing labels stabilizes the loss landscape for the AdamW optimizer
    all_labels = torch.tensor([dataset[i][1].item() for i in range(len(dataset))])
    _, target_mean, target_std = normalize_targets(all_labels)

    # 4. Model, Optimizer, and Loss Initialization
    model = DinoRegressorPRO().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.L1Loss() # MAE is more robust than MSE for this regression task

    # 5. Tracking Variables for Convergence & Checkpointing
    best_val_mae = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    train_losses = []
    val_maes = []

    print(f"--- Starting Convergence Training on {DEVICE} ---")

    for epoch in range(MAX_EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        running_epoch_loss = 0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            # Target normalization to standardized units
            labels_norm = (labels - target_mean) / target_std

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels_norm)
            loss.backward()
            optimizer.step()
            
            running_epoch_loss += loss.item()

        train_losses.append(running_epoch_loss)

        # --- VALIDATION PHASE (Early Stopping Logic) ---
        if (epoch + 1) % VALIDATION_FREQ == 0:
            model.eval()
            current_val_mae_sum = 0
            with torch.no_grad():
                for imgs, labels, _ in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    
                    # Inference & Denormalization to human-readable units (pallet count)
                    preds_norm = model(imgs)
                    preds = preds_norm * target_std + target_mean
                    
                    current_val_mae_sum += torch.abs(preds - labels).item()
            
            avg_val_mae = current_val_mae_sum / len(val_loader)
            val_maes.append(avg_val_mae)
            
            print(f"Epoch {epoch+1:04d} | Loss: {running_epoch_loss:.4f} | Val MAE: {avg_val_mae:.4f}")

            # Improvement Check
            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                best_model_wts = copy.deepcopy(model.state_dict()) # Save best weights in memory
                epochs_no_improve = 0
                print("  >> New Global Minimum Detected. Checkpoint updated.")
            else:
                epochs_no_improve += 1
        else:
            val_maes.append(None) # Placeholders for consistent plotting indices

        # Termination Criteria: Stop when model reaches a performance plateau
        if epochs_no_improve >= PATIENCE:
            print(f"\nConvergence reached. Early stopping triggered at epoch {epoch+1}.")
            break

    # 6. Post-Training: Final Parameter Restoration
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "best_pallet_regressor_final.pth")
    print(f"Process Complete. Optimal Validation MAE: {best_val_mae:.4f}")
    
    return train_losses, val_maes, test_ds, target_mean, target_std


def run_project():
    """
    Complete project execution: Data prep, Training with Early Stopping, 
    and Professional Metric Visualization.
    """
    
    # --- Configuration ---
    MAX_EPOCHS = 10000    
    PATIENCE = 50         
    VALIDATION_FREQ = 5   
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
    ])

    # 2. Dataset Partitioning
    dataset = PalletMultiViewDataset("./DataSet", transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1)
    
    # 3. Target Scaling (Z-score)
    all_labels = torch.tensor([dataset[i][1].item() for i in range(len(dataset))])
    _, target_mean, target_std = normalize_targets(all_labels)

    # 4. Model Initialization
    model = DinoRegressorPRO().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.L1Loss() 

    # 5. Tracking Convergence
    best_val_mae = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    train_losses = []
    val_maes = []

    print(f"--- Initiating Training: Searching for Global Minimum on {DEVICE} ---")

    for epoch in range(MAX_EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        running_epoch_loss = 0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            labels_norm = (labels - target_mean) / target_std

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels_norm)
            loss.backward()
            optimizer.step()
            running_epoch_loss += loss.item()

        train_losses.append(running_epoch_loss)

        # --- VALIDATION PHASE & EARLY STOPPING ---
        if (epoch + 1) % VALIDATION_FREQ == 0:
            model.eval()
            current_val_mae_sum = 0
            with torch.no_grad():
                for imgs, labels, _ in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    preds = model(imgs) * target_std + target_mean
                    current_val_mae_sum += torch.abs(preds - labels).item()
            
            avg_val_mae = current_val_mae_sum / len(val_loader)
            val_maes.append(avg_val_mae)
            
            print(f"Epoch {epoch+1:04d} | Loss: {running_epoch_loss:.4f} | Val MAE: {avg_val_mae:.4f}")

            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                print("  >> Global Minimum Update: Checkpoint Saved.")
            else:
                epochs_no_improve += 1
        else:
            val_maes.append(None) 

        if epochs_no_improve >= PATIENCE:
            print(f"\nConvergence threshold reached at epoch {epoch+1}.")
            break

    # 6. Restoration & Final Export
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "best_pallet_regressor.pth")
    
    # 7. AUTOMATED VISUAL REPORTING
    print("\nGenerating Final Performance Report...")
    plot_metrics(train_losses, val_maes)
    
    return train_losses, val_maes
