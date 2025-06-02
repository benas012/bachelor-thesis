# 1. Imports
import os, re, glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score

# 2. Labeling function
def categorize_ad(ctr, impressions):
    if impressions < 100_000 and ctr < 0.1:
        return 0
    elif impressions >= 300_000 or ctr >= 0.2:
        return 2
    else:
        return 1

# 3. Dataset
class ViralAdsDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.transform = transform
        self.samples = []
        pattern = re.compile(r"^([\d.]+)_([\d]+)_.+\.jpg$")
        for path in glob.glob(os.path.join(folder, "*.jpg")):
            fn = os.path.basename(path)
            m = pattern.match(fn)
            if not m: continue
            ctr, impr = float(m.group(1)), float(m.group(2))
            label = categorize_ad(ctr, impr)
            self.samples.append((path, label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# 4. Transforms with strong augmentations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

# 5. Prepare dataset and loaders
dataset = ViralAdsDataset(r"C:\Bakalauras\filtered_images", transform=None)
n_train = int(0.8*len(dataset))
n_val   = len(dataset) - n_train
train_ds, val_ds = random_split(dataset, [n_train, n_val])
# assign transforms
train_ds.dataset.transform = train_transform
val_ds.dataset.transform   = val_transform

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

# 6. Model setup (ResNet-18 fine-tune head only)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
# freeze all except fc
for name, param in model.named_parameters():
    if not name.startswith("fc."):
        param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# 7. Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}  Train Loss: {epoch_loss:.4f}")

# 8. Evaluation
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

print(classification_report(y_true, y_pred, digits=4))

mae  = mean_absolute_error(y_true, y_pred)
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_true, y_pred)

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")
