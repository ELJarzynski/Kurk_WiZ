import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import math

# --- PARAMETRY ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
NUM_VIEWS = 8
BATCH_SIZE = 4 
EPOCHS = 100
LEARNING_RATE = 1e-4

# 1. DATASET: Obsługa struktury DataSet/Set*/
class PalletMultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Szukamy folderów Set1, Set2 itd. wewnątrz DataSet
        self.scenes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_name = self.scenes[idx]
        scene_path = os.path.join(self.root_dir, scene_name)
        
        # Pobieramy obrazy i sortujemy je
        img_names = sorted([f for f in os.listdir(scene_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        if len(img_names) < NUM_VIEWS:
            # Jeśli zdjęć jest więcej, bierzemy pierwsze 8. Jeśli mniej - błąd.
            raise ValueError(f"Folder {scene_name} ma tylko {len(img_names)} zdjęć! Wymagane {NUM_VIEWS}.")

        images = []
        for i in range(NUM_VIEWS):
            img = Image.open(os.path.join(scene_path, img_names[i])).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
            
        images = torch.stack(images) # [NUM_VIEWS, 3, 224, 224]
        
        # LOGIKA ETYKIETY: "1_1_1809" -> bierzemy ostatnią część jako wartość do regresji
        try:
            label_value = float(scene_name.split('_')[-1])
        except ValueError:
            label_value = 0.0 # Backup
            
        return images, torch.tensor([label_value], dtype=torch.float32)

# 2. MODEL: DINOv2 z fuzją widoków
class PalletRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # Ładowanie DINOv2 (ViT-S/14)
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # Zamrożenie wag backbone'u dla małych zbiorów danych
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.embed_dim = self.backbone.embed_dim # 384
        
        self.regressor = nn.Sequential(
            nn.Linear(self.embed_dim * NUM_VIEWS, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # x shape: [Batch, Views, C, H, W]
        b, v, c, h, w = x.shape
        x = x.view(b * v, c, h, w) 
        
        features = self.backbone(x) # [b*v, 384]
        
        # Łączymy cechy ze wszystkich 8 zdjęć w jeden długi wektor
        features = features.view(b, v * self.embed_dim) 
        return self.regressor(features)

# 3. URUCHOMIENIE
def run_project():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Upewnij się, że ścieżka prowadzi do głównego folderu "DataSet"
    full_dataset = PalletMultiViewDataset(root_dir='./DataSet', transform=transform)
    
    if len(full_dataset) < 2:
        print("Za mało danych w folderze DataSet!")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    model = PalletRegressor().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Przy regresji MSE często lepiej zbiega niż L1

    print(f"Start: {len(full_dataset)} scen, Urządzenie: {DEVICE}")

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Walidacja co 5 epok
        if (epoch + 1) % 5 == 0:
            model.eval()
            mae = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    preds = model(imgs)
                    mae += torch.abs(preds - labels).item()
            
            avg_mae = mae / len(val_loader)
            print(f"Epoka {epoch+1:03d} | Loss: {total_train_loss/len(train_loader):.4f} | Val MAE: {avg_mae:.2f}")

if __name__ == "__main__":
    run_project()
