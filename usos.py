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




class PalletMultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Szukamy folderów Set1, Set2... wewnątrz DataSet
        self.scenes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        print(f"\n--- INICJALIZACJA DATASETU ---")
        print(f"Znaleziono {len(self.scenes)} folderów (scen): {self.scenes}")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_name = self.scenes[idx]
        scene_path = os.path.join(self.root_dir, scene_name)
        
        # Pobieramy pliki .jpg/.png
        img_names = sorted([f for f in os.listdir(scene_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        if len(img_names) < NUM_VIEWS:
            print(f"!!! BŁĄD: Folder {scene_name} ma tylko {len(img_names)} zdjęć!")
            return torch.zeros((NUM_VIEWS, 3, 224, 224)), torch.tensor([0.0])

        # Pobieramy etykietę z nazwy pierwszego zdjęcia (np. 1_1_1809.jpg)
        first_img = img_names[0]
        try:
            # Rozbijamy: "1_1_1809.jpg" -> "1_1_1809" -> ["1", "1", "1809"]
            label_value = float(first_img.split('.')[0].split('_')[2])
        except Exception as e:
            print(f"!!! BŁĄD PARSOWANIA w {scene_name} plik {first_img}: {e}")
            label_value = -1.0

        # PRINT DEBUGUJĄCY - to pokaże Ci co model bierze w każdej iteracji
        # print(f"DEBUG: Scena {scene_name} | Plik: {first_img} | WYKRYTO PALET: {label_value}")

        images = []
        for i in range(NUM_VIEWS):
            img = Image.open(os.path.join(scene_path, img_names[i])).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
            
        return torch.stack(images), torch.tensor([label_value], dtype=torch.float32)

def run_project():
    # 1. Preprocessing dla DINOv2
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Inicjalizacja Datasetu
    # Upewnij się, że ścieżka './DataSet' jest poprawna
    full_dataset = PalletMultiViewDataset(root_dir='./DataSet', transform=transform)
    
    num_total = len(full_dataset)
    if num_total < 10:
        print(f"UWAGA: Masz tylko {num_total} zestawów. Kod jest ustawiony pod 10.")

    # 3. MANUALNY PODZIAŁ (Indeksy: 0-7 trening, 8 walidacja, 9 test)
    indices = list(range(num_total))
    train_indices = indices[0:8]
    val_indices   = indices[8:9]
    test_indices  = indices[9:10]

    print("\n" + "="*40)
    print(f"PODZIAŁ DANYCH:")
    print(f"TRENING (8):    {[full_dataset.scenes[i] for i in train_indices]}")
    print(f"WALIDACJA (1):  {[full_dataset.scenes[i] for i in val_indices]}")
    print(f"TEST (1):       {[full_dataset.scenes[i] for i in test_indices]}")
    print("="*40 + "\n")

    train_ds = torch.utils.data.Subset(full_dataset, train_indices)
    val_ds   = torch.utils.data.Subset(full_dataset, val_indices)
    test_ds  = torch.utils.data.Subset(full_dataset, test_indices)

    # 4. DataLoadery
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1)
    test_loader  = DataLoader(test_ds, batch_size=1)

    # 5. Model, Optimizer, Criterion
    model = PalletRegressor().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() 

    # 6. PĘTLA TRENINGOWA
    print(f"Rozpoczynam trening na {DEVICE}...")
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

        # Walidacja co 10 epok
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_mae = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    preds = model(imgs)
                    val_mae += torch.abs(preds - labels).item()
            
            print(f"Epoka {epoch+1:03d}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val MAE: {val_mae/len(val_loader):.2f}")

    # 7. FINALNY TEST (Wykorzystanie TEST_LOADERA)
    print("\n" + "!"*40)
    print("URUCHAMIAM TEST KOŃCOWY (ZBIÓR TESTOWY)")
    print("!"*40)
    
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs)
            
            poczatek_sceny = full_dataset.scenes[test_indices[0]]
            y_pred = preds.item()
            y_true = labels.item()
            
            print(f"WYNIK DLA FOLDERU: {poczatek_sceny}")
            print(f"  -> PRZEWIDZIANO: {y_pred:.2f} palet")
            print(f"  -> PRAWDA:       {y_true:.2f} palet")
            print(f"  -> RÓŻNICA:      {abs(y_pred - y_true):.2f}")
    
    print("\nTrening i test zakończony.")

if __name__ == "__main__":
    run_project()
