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







def run_project():
    # 1. Pre-processing Pipeline
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # Lista nazw Twoich podfolderów (Dataset1, Dataset2, itd.)
    # Zakładam, że są one bezpośrednio w katalogu "./DataSet"
    dataset_names = [f"Dataset{i}" for i in range(1, 11)]
    results = {}

    # Główna pętla Cross-Validation (10 iteracji)
    for test_folder in dataset_names:
        print(f"\n--- Ewaluacja na zbiorze: {test_folder} ---")
        
        # Tworzymy listy folderów treningowych (wszystkie poza testowym)
        train_folders = [d for d in dataset_names if d != test_folder]

        # 2. Przygotowanie danych (wymaga modyfikacji klasy Dataset, aby filtrowała po folderach)
        # Zakładam, że PalletMultiViewDataset przyjmuje argument subfolders
        train_ds = PalletMultiViewDataset("./DataSet", subfolders=train_folders, transform=transform)
        test_ds  = PalletMultiViewDataset("./DataSet", subfolders=[test_folder], transform=transform)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_ds, batch_size=1)

        # Statystyki celu (Z-score) dla zbioru treningowego
        all_train_labels = torch.tensor([train_ds[i][1].item() for i in range(len(train_ds))])
        _, mean, std = normalize_targets(all_train_labels)

        # 3. Model & Optimization Setup (Reset modelu w każdej iteracji!)
        model = DinoRegressorPRO().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        criterion = nn.L1Loss()

        # 4. Pętla Treningowa
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for imgs, labels, _ in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                labels_norm = (labels - mean) / std

                optimizer.zero_grad()
                pred_norm = model(imgs)
                loss = criterion(pred_norm, labels_norm)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"Dataset {test_folder} | Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

        # 5. Ewaluacja finalna dla danego folderu
        model.eval()
        total_mae = 0
        with torch.no_grad():
            for imgs, labels, _ in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                pred = model(imgs) * std + mean
                total_mae += abs(pred.item() - labels.item())
        
        avg_mae = total_mae / len(test_loader)
        results[test_folder] = avg_mae
        print(f"ZAKOŃCZONO: Błąd (MAE) dla {test_folder}: {avg_mae:.4f}")

    # 6. Podsumowanie wyników
    print("\n" + "="*30)
    print("FINALNE WYNIKI DLA KAŻDEGO DATASETU:")
    for ds_name, error in results.items():
        print(f"{ds_name}: MAE = {error:.4f}")
    
    avg_total_mae = sum(results.values()) / len(results)
    print(f"\nŚredni błąd ze wszystkich prób: {avg_total_mae:.4f}")
