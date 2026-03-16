import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import math

# --- PARAMETRY POD NVIDIA T4 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224          # Wielokrotność 14 dla DINOv2 
NUM_VIEWS = 8           # 8 zdjęć na magazyn
BATCH_SIZE = 4          # Optymalne dla T4 (8 zdjęć * 4 = 32 obrazy na batch) [1]
EPOCHS = 100            # Przy 10 zestawach potrzebujemy więcej epok na zbieżność
LEARNING_RATE = 1e-4    # Niskie LR dla Transformerów [2, 3]

# 1. DATASET: Ładowanie 8 ujęć jako jedna scena
class PalletMultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Szukamy tylko folderów
        self.scenes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_name = self.scenes[idx]
        scene_path = os.path.join(self.root_dir, scene_name)
        
        # Pobieramy pliki graficzne, sortujemy dla spójności ujęć
        img_names = sorted([f for f in os.listdir(scene_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        if len(img_names) < NUM_VIEWS:
            raise ValueError(f"Folder {scene_name} ma za mało zdjęć! Wymagane {NUM_VIEWS}.")

        images =
        for i in range(NUM_VIEWS):
            img = Image.open(os.path.join(scene_path, img_names[i])).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
            
        # Tensor: 
        images = torch.stack(images)
        
        # Etykieta z nazwy folderu: "magazyn_1_count_12" -> 12.0 [4, 5]
        label = float(scene_name.split('_')[-1])
        return images, torch.tensor([label], dtype=torch.float32)

# 2. MODEL: DINOv2 z głowicą regresyjną
class PalletRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # DINOv2 wyciąga lepsze cechy geometryczne niż zwykły ViT 
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # Zamrażamy backbone, bo przy 10 zestawach danych szybko go "zepsujesz" (overfitting)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.embed_dim = self.backbone.embed_dim # 384 dla vits14
        
        # Agregator widoków i regresja liniowa [6, 4]
        self.regressor = nn.Sequential(
            nn.Linear(self.embed_dim * NUM_VIEWS, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1) # Wynik: liczba palet
        )

    def forward(self, x):
        b, v, c, h, w = x.shape
        x = x.view(b * v, c, h, w) # Spłaszczamy batch dla backbone'u
        
        features = self.backbone(x) # [b*8, 384]
        
        features = features.view(b, v * self.embed_dim) # Fuzja widoków [b, 8*384]
        return self.regressor(features)

# 3. METRYKI I TRENING
def run_project():
    # Preprocessing (ImageNet stats dla DINOv2) 
    transform = transforms.Compose(, std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = PalletMultiViewDataset(root_dir='./data', transform=transform)
    
    # Podział na train/val (np. 8 scen trening, 2 sceny walidacja)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    model = PalletRegressor().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss() # MAE - najlepsza do liczenia sztuk 

    print(f"Trening na T4 | Rozmiar batcha: {BATCH_SIZE} | Próbki: {len(full_dataset)}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # EWALUACJA I METRYKI
        model.eval()
        mae, mse = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs)
                mae += torch.abs(preds - labels).item()
                mse += torch.pow(preds - labels, 2).item()
        
        val_mae = mae / len(val_loader)
        val_rmse = math.sqrt(mse / len(val_loader))

        if (epoch + 1) % 10 == 0:
            print(f"Epoka {epoch+1:03d} | Train MAE: {train_loss/len(train_loader):.2f} | Val MAE: {val_mae:.2f} | Val RMSE: {val_rmse:.2f}")

    print("\nTrening zakończony. Model potrafi oszacować liczbę palet.")

if __name__ == "__main__":
    run_project()
