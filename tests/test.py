# ...existing code...
import itertools
import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

# Ignore Warnings
import warnings

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sns.set_style('darkgrid')
warnings.filterwarnings("ignore")

print('modules loaded')


data_dir = './../bloodcells_dataset'
filepaths = []
labels = []

folds = [f for f in os.listdir(data_dir) if not f.startswith('.')]  # Skip hidden files
for fold in folds:
    foldpath = os.path.join(data_dir, fold)
    # Skip if not a directory or if it's one of the excluded classes
    if not os.path.isdir(foldpath) or fold in ['ig', 'neutrophil']:
        continue
    filelist = [f for f in os.listdir(foldpath) if not f.startswith('.')]  # Skip hidden files
    for file in filelist:
        fpath = os.path.join(foldpath, file)
        if os.path.isfile(fpath):  # Ensure it's a file
            filepaths.append(fpath)
            labels.append(fold)

# Create dataframe
Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='labels')
df = pd.concat([Fseries, Lseries], axis=1)

df.head()
df.shape
df.info()

labelscount = df['labels'].value_counts().reset_index()
labelscount.columns = ['labels', 'count']

plt.figure(figsize=(10, 6))
sns.barplot(x='labels', y='count', data=labelscount, palette='viridis')
plt.title('Count of Each Label')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()
df['labels'].value_counts().plot(kind='bar', color='skyblue')
plt.show()

train_df, dummy_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=43)
valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=43)

batch_size = 16
img_size = (224, 224)
channels = 3
class_names = sorted(train_df['labels'].unique())
class_to_idx = {c: i for i, c in enumerate(class_names)}

# Define custom dataset
class BloodCellsDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, 'filepaths']
        label_str = self.dataframe.loc[idx, 'labels']
        label_idx = class_to_idx[label_str]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label_idx

# Define transforms
train_transform = T.Compose([
    T.Resize(img_size),
    T.ToTensor()
])

test_transform = T.Compose([
    T.Resize(img_size),
    T.ToTensor()
])

# Create datasets
train_dataset = BloodCellsDataset(train_df, transform=train_transform)
valid_dataset = BloodCellsDataset(valid_df, transform=test_transform)
test_dataset = BloodCellsDataset(test_df, transform=test_transform)

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=len(class_names)):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128 * (img_size[0] // 8) * (img_size[1] // 8), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(class_names)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training function
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

# Validation function
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss / total, 100.0 * correct / total

# Train and validate
num_epochs = 5
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("Starting training loop...")
for epoch in range(num_epochs):
  print(f"\nEpoch {epoch+1}/{num_epochs}")
  print("-" * 50)
  
  print("Training phase...")
  start_time = time.time()
  train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
  train_time = time.time() - start_time
  
  print("Validation phase...")
  start_time = time.time()
  val_loss, val_acc = validate(model, valid_loader, criterion)
  val_time = time.time() - start_time
  
  # Store metrics
  train_losses.append(train_loss)
  val_losses.append(val_loss)
  train_accs.append(train_acc)
  val_accs.append(val_acc)

  print(f"Results for epoch {epoch+1}:")
  print(f"Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}% (took {train_time:.2f}s)")
  print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}% (took {val_time:.2f}s)")
  
  if epoch > 0:
    print("Change from previous epoch:")
    print(f"Training   - Loss: {train_loss - train_losses[-2]:.4f}, Accuracy: {train_acc - train_accs[-2]:.2f}%")
    print(f"Validation - Loss: {val_loss - val_losses[-2]:.4f}, Accuracy: {val_acc - val_accs[-2]:.2f}%")

print("\nTraining completed!")

# Plot training history
Epochs = range(1, num_epochs+1)
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(Epochs, train_losses, 'r', label='Training loss')
plt.plot(Epochs, val_losses, 'g', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(Epochs, train_accs, 'r', label='Training Accuracy')
plt.plot(Epochs, val_accs, 'g', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.show()

# Evaluate on train, valid, test sets
def evaluate_loader(model, loader, criterion):
    loss, acc = validate(model, loader, criterion)
    return loss, acc

print('-' * 20)
train_loss, train_acc = evaluate_loader(model, train_loader, criterion)
print(f"Train Loss: {train_loss:.4f}\nTrain Accuracy: {train_acc:.2f}%")
print('-' * 20)
valid_loss, valid_acc = evaluate_loader(model, valid_loader, criterion)
print(f"Validation Loss: {valid_loss:.4f}\nValidation Accuracy: {valid_acc:.2f}%")
print('-' * 20)
test_loss, test_acc = evaluate_loader(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}\nTest Accuracy: {test_acc:.2f}%")

# Predictions and confusion matrix
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment='center',
             color='white' if cm[i, j] > thresh else 'black')
plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print(classification_report(all_labels, all_preds, target_names=class_names))

# Save the model with versioning
base_model_name = 'bloodcell_model'
version = 1
model_save_path = f'{base_model_name}_v{version}.pth'

# Check if file exists and increment version number
while os.path.exists(model_save_path):
  print(f"Model {model_save_path} already exists! \n incrementing version number...")
  version += 1
  model_save_path = f'{base_model_name}_v{version}.pth'

# Save just the model state
model_save_path = f'{base_model_name}_v{version}.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Save training data and metadata separately
training_data = {
  'optimizer_state_dict': optimizer.state_dict(),
  'epoch': num_epochs,
  'train_losses': train_losses,
  'val_losses': val_losses,
  'train_accs': train_accs,
  'val_accs': val_accs,
  'class_names': class_names,
  'class_to_idx': class_to_idx
}
training_save_path = f'{base_model_name}_training_v{version}.pth'
torch.save(training_data, training_save_path)
print(f"Training data saved to {training_save_path}")
