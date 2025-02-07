import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from logger import create_log_file, write_log  # Import modul logger
import time  # Import untuk menghitung waktu

log_file = create_log_file('training_log')
# Konfigurasi awal
# train_dir = '../../../.venv/dataset/dataset_cgrabcut_final/Train_augmented/'
train_dir = '../../../.venv/dataset/dataset_fix_final/Train_augmented/'
# train_dir = '../../../.venv/dataset/dataset_nosegmentation_new/Train_augmented/'
# train_dir = '../../../.venv/dataset/dataset_fix_final/Train/'
# train_dir = '../../../.venv/dataset/dataset_nosegmentation_new/Train/'
# valid_dir = '../../../.venv/dataset/dataset_cgrabcut_final/Valid/'
valid_dir = '../../../.venv/dataset/dataset_fix_final/Valid/'
# valid_dir = '../../../.venv/dataset/dataset_nosegmentation_new/Valid/'
# model_save_path = '../../../.venv/model/model_mobilenetv2_exp32_f1_full25epoch.pth'
model_save_path = '../../../.venv/model/model_test_aja.pth'

num_classes = 3
batch_size = 64
epochs = 25
patience = 5
learning_rate = 0.0001

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


# Print the number of samples in each dataset
num_train_samples = len(train_dataset)
num_valid_samples = len(valid_dataset)

print(f'Number of training samples: {num_train_samples}')
print(f'Number of validation samples: {num_valid_samples}')

# Logging the dataset sizes
dataset_size_message = (f"Number of training samples: {num_train_samples}\n"
                        f"Number of validation samples: {num_valid_samples}")
write_log(log_file, dataset_size_message)

# Load MobileNetV2 pretrained model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Freeze all layers
for param in model.features.parameters():
    param.requires_grad = False

# Hitung total parameter di backbone dan unfreeze 5%
total_layers = len(list(model.features.parameters()))
unfreeze_count = int(0.5 * total_layers)

# Tampilkan informasi tentang parameter
print(f"Total layers in model.features: {total_layers}")
print(f"Number of layers to unfreeze: {unfreeze_count}")

# Unfreeze 5% parameter terakhir
for param in list(model.features.parameters())[-unfreeze_count:]:
    param.requires_grad = True

# Tampilkan parameter yang di-unfreeze
for name, param in model.features.named_parameters():
    if param.requires_grad:
        print(f"Unfrozen parameter: {name}")


# Logging parameter yang di-unfreeze
write_log(log_file, "Unfrozen parameters:")
for name, param in model.features.named_parameters():
    if param.requires_grad:
        write_log(log_file, f" - {name}")

# Sebelum fully connected layer
feature_extractor = model.features
dummy_input = torch.randn(1, 3, 224, 224)  # Input contoh
output = feature_extractor(dummy_input)
print("Shape setelah feature extractor:", output.shape)  # Output: [1, 1280, 7, 7]


# Modify classifier
model.classifier[1] = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, num_classes)
)


# Setelah melalui layer fully connected pertama
# fc_layer = nn.Linear(1280 * 7 * 7, 256)
# flattened_output = fc_layer(output.view(output.size(0), -1))  # Secara manual diratakan
# print("Shape setelah fully connected:", flattened_output.shape)  # Output: [1, 256]

print("Model Architecture:")
print(model)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4) #default betas=(0.9, 0.999), eps=1e-8

# Early stopping variables
best_f1 = 0.0
best_precision = 0.0
best_recall = 0.0
best_accuracy = 0.0
best_epoch = 0
epochs_no_improve = 0
early_stop = False

# Training and validation history
train_loss_history = []
valid_loss_history = []
train_acc_history = []
valid_acc_history = []

# Mencatat waktu mulai pelatihan
start_time = time.time()

# Training loop
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')

    # Training phase
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc.item())
    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # Logging training results
    train_log_message = (f'Epoch {epoch + 1}/{epochs}, '
                         f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}')
    write_log(log_file, train_log_message)


    # Validation phase
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(valid_loader.dataset)
    epoch_acc = running_corrects.double() / len(valid_loader.dataset)
    valid_loss_history.append(epoch_loss)
    valid_acc_history.append(epoch_acc.item())

    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

    # Logging validation results
    valid_log_message = (f'Epoch {epoch + 1}/{epochs}, '
                         f'Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_acc:.4f}, '
                         f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
    write_log(log_file, valid_log_message)


    # Save best model based on F1-score
    if f1 > best_f1:
        best_f1 = f1
        best_precision = precision
        best_recall = recall
        best_accuracy = epoch_acc.item()
        best_epoch = epoch + 1
        epochs_no_improve = 0
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at epoch {epoch + 1}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            early_stop = True
            break

# Menghitung total waktu pelatihan
end_time = time.time()
total_training_time = end_time - start_time
hours, rem = divmod(total_training_time, 3600)
minutes, seconds = divmod(rem, 60)

# Tampilkan waktu pelatihan
print(f'Total Training Time: {int(hours)}h {int(minutes)}m {int(seconds)}s')
write_log(log_file, f'Total Training Time: {int(hours)}h {int(minutes)}m {int(seconds)}s')


# Plotting accuracy and loss
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(train_acc_history, label='Train Accuracy', color='blue', linestyle='-', marker='o')
plt.plot(valid_acc_history, label='Validation Accuracy', color='green', linestyle='--', marker='x')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(train_loss_history, label='Train Loss', color='red', linestyle='-', marker='o')
plt.plot(valid_loss_history, label='Validation Loss', color='orange', linestyle='--', marker='x')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.grid(True)

plt.tight_layout()
plt.show()

# Summary of best model
print(f'Best Model at Epoch {best_epoch}:')
print(f'Accuracy: {best_accuracy:.4f}')
print(f'Precision: {best_precision:.4f}')
print(f'Recall: {best_recall:.4f}')
print(f'F1-Score: {best_f1:.4f}')
write_log(log_file, f'Best Model at Epoch {best_epoch} - Accuracy: {best_accuracy:.4f}, '
                    f'Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1-Score: {best_f1:.4f}')

