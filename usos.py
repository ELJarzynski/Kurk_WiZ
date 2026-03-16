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

# 1. DATASET: Ładowanie 8 ujęć z folderów Set* w DataSet/
class PalletMultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Szukamy folderów (scen) w DataSet i sortujemy je
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Nie znaleziono folderu: {root_dir}")
        self.scenes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_name = self.scenes[idx]
        scene_path = os.path.join(self.root_dir, scene_name)
        
        img_names = sorted([f for f in os.listdir(scene_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        if len(img_names) < NUM_VIEWS:
            raise ValueError(f"Folder {scene_name} ma za mało zdjęć ({len(img_names)})! Wymagane {NUM_VIEWS}.")

        images = []
        for i in range(NUM_VIEWS):
            img = Image.open(os.path.join(scene_path, img_names[i])).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
            
        images = torch.stack(images)
        
        # Pobieranie liczby palet z nazwy folderu (np. "1_1_1809" -> 1809.0)
        try:
            label_value = float(scene_name.split('_')[-1])
        except (ValueError, IndexError):
            label_value = 0.0
            
        return images, torch.tensor([label_value], dtype=torch.float32)

# 2. MODEL: DINOv2 z poprawionym kształtem tensorów (reshape)
class PalletRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # Zamrożenie backbone'u
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
        b, v, c, h, w = x.shape
        # Używamy reshape zamiast view, aby uniknąć błędu "contiguous"
        x = x.reshape(b * v, c, h, w) 
        
        features = self.backbone(x)
        
        # Łączymy cechy ze wszystkich 8 ujęć
        features = features.reshape(b, v * self.embed_dim) 
        return self.regressor(features)

# 3. FUNKCJA ANALIZY BŁĘDÓW
def check_pallet_errors(model, loader, device, title="WALIDACJA"):
    model.eval()
    print(f"\n--- {title}: PORÓWNANIE W SZTUKACH ---")
    print(f"{'Nr':<3} | {'Prawda':>10} | {'Predykcja':>10} | {'Błąd (szt)':>10}")
    print("-" * 45)

    total_error = 0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            
            real = labels.item()
            pred = outputs.item()
            diff = abs(real - pred)
            total_error += diff
            
            print(f"{i+1:<3} | {real:10.2f} | {pred:10.2f} | {diff:10.2f}")

    avg_err = total_error / len(loader)
    print("-" * 45)
    print(f"Średni błąd w tej grupie: {avg_err:.2f} palety\n")

# 4. GŁÓWNA PĘTLA
def run_project():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # UWAGA: Upewnij się, że folder nazywa się dokładnie "DataSet"
    full_dataset = PalletMultiViewDataset(root_dir='./DataSet', transform=transform)
    
    # Podział na train i val (8 folderów trening, 2 walidacja)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    model = PalletRegressor().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() 

    print(f"Start: {len(full_dataset)} scen, Urządzenie: {DEVICE}")

    # Trening
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

        if (epoch + 1) % 10 == 0:
            print(f"Epoka {epoch+1:03d}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f}")

    # --- FINALNA EWALUACJA ---
    # Sprawdzamy na danych, których model NIE widział
    check_pallet_errors(model, val_loader, DEVICE, title="WYNIKI WALIDACJI")
    
    # Opcjonalnie: sprawdźmy też na treningowych, żeby zobaczyć czy model je zapamiętał
    # check_pallet_errors(model, DataLoader(train_ds, batch_size=1), DEVICE, title="WYNIKI TRENINGOWE")

if __name__ == "__main__":
    run_project()
