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
BATCH_SIZE = 2         # Zmniejszone dla T4 przy odblokowanym backbone
EPOCHS = 100
LEARNING_RATE = 2e-5   # Niższe LR, bo trenujemy cały model (fine-tuning)
SCALE_FACTOR = 1000.0  # Skalujemy etykiety: 2219 -> 2.219

# 1. DATASET: Obsługa struktury DataSet/Set*/ z wyciąganiem etykiety z nazwy pliku
class PalletMultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Sortujemy foldery Set1, Set2...
        self.scenes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))], 
                             key=lambda x: int(x.replace('Set', '')) if x.replace('Set', '').isdigit() else x)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_name = self.scenes[idx]
        scene_path = os.path.join(self.root_dir, scene_name)
        
        img_names = sorted([f for f in os.listdir(scene_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        if len(img_names) < NUM_VIEWS:
            raise ValueError(f"Folder {scene_name} ma za mało zdjęć!")

        # Etykieta z nazwy pierwszego pliku: np. "1_1_2219.jpg" -> 2219
        first_img = img_names[0]
        try:
            # name_parts[2] to ilość palet
            label_raw = float(first_img.split('.')[0].split('_')[2])
            label_value = label_raw / SCALE_FACTOR  # Skalowanie do zakresu ~2.0
        except:
            label_value = 0.0

        images = []
        for i in range(NUM_VIEWS):
            img = Image.open(os.path.join(scene_path, img_names[i])).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
            
        return torch.stack(images), torch.tensor([label_value], dtype=torch.float32), scene_name

# 2. MODEL: DINOv2 z odblokowanymi wagami
class PalletRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        # Odblokowujemy wagi - przy 10 zestawach to ryzykowne (overfitting), 
        # ale przy tak dużym błędzie model MUSI się bardziej dostosować.
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        self.embed_dim = self.backbone.embed_dim
        self.regressor = nn.Sequential(
            nn.Linear(self.embed_dim * NUM_VIEWS, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        b, v, c, h, w = x.shape
        x = x.reshape(b * v, c, h, w) 
        features = self.backbone(x) 
        features = features.reshape(b, v * self.embed_dim) 
        return self.regressor(features)

# 3. TRENING I TEST
def run_project():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Augmentacja dla małej ilości danych
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = PalletMultiViewDataset(root_dir='./DataSet', transform=transform)
    
    # Podział: 8 trenujemy, 1 walidujemy, 1 testujemy
    train_ds, val_ds, test_ds = random_split(dataset, [8, 1, 1], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)
    test_loader = DataLoader(test_ds, batch_size=1)

    model = PalletRegressor().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.HuberLoss() # Stabilniejszy niż MSE dla dużych wartości

    print(f"Rozpoczynam trening na {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            val_mae = 0
            with torch.no_grad():
                for imgs, labels, _ in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    preds = model(imgs)
                    # Powrót do realnej skali dla MAE
                    val_mae += torch.abs(preds*SCALE_FACTOR - labels*SCALE_FACTOR).item()
            
            print(f"Epoka {epoch+1:03d} | Loss: {train_loss/len(train_loader):.4f} | Val MAE: {val_mae/len(val_loader):.2f} szt.")

    # --- TEST KOŃCOWY ---
    print("\n" + "="*30)
    print("WYNIKI TESTU KOŃCOWEGO")
    print("="*30)
    model.eval()
    with torch.no_grad():
        for imgs, labels, name in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            pred = model(imgs)
            
            real_pred = pred.item() * SCALE_FACTOR
            real_true = labels.item() * SCALE_FACTOR
            
            print(f"Folder: {name[0]}")
            print(f" -> PRZEWIDZIANO: {real_pred:.2f} palet")
            print(f" -> PRAWDA:       {real_true:.2f} palet")
            print(f" -> RÓŻNICA:      {abs(real_pred - real_true):.2f}")
    print("="*30)

if __name__ == "__main__":
    run_project()
