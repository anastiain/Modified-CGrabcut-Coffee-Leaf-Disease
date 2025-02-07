import torch
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformasi yang sama dengan saat training
test_transforms = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset testing
# test_dir = '../../../.venv/dataset/dataset_fix_nosegmentation/Test/' # Path ke dataset test
test_dir = '../../../.venv/dataset/dataset_fix_final/Test/'  # Path ke dataset test
# test_dir = '../../../.venv/dataset/dataset_cgrabcut_final/Test/'
# test_dir = '../../../.venv/dataset/dataset_nosegmentation_new/Test/'  # Path ke dataset test
# test_dir = '../../../.venv/dataset/dataset_grabcut_final/Test/'  # Path ke dataset test
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
num_classes = 3

# Load the model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
# model.classifier[1] = nn.Sequential(
#     nn.Dropout(0.5),
#     nn.Linear(model.classifier[1].in_features, 128),
#     nn.ReLU(),
#     nn.BatchNorm1d(128),
#     # nn.Dropout(0.3),
#     nn.Linear(128, 3),
# )

model.classifier[1] = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 256),  # Linear layer pertama / DENSE LAYER
    nn.BatchNorm1d(256),                             # BatchNorm untuk normalisasi output dari linear
    nn.ReLU(),                                       # ReLU sebagai fungsi aktivasi
    nn.Dropout(0.3),                                 # Dropout pertama setelah aktivasi
    nn.Linear(256, 128),        # Linear layer tambahan (opsional, jika ingin lebih banyak Dropout)
    nn.ReLU(),                                       # ReLU untuk non-linearitas tambahan
    nn.Dropout(0.3),                                 # Dropout kedua setelah aktivasi lagi
    nn.Linear(128, num_classes),           # Linear layer untuk output sesuai jumlah kelas
    # nn.LogSoftmax(dim=1)                             # LogSoftmax di akhir untuk klasifikasi
)

# model.load_state_dict(torch.load('../../../.venv/model/model_test_aja.pth'))
# model.load_state_dict(torch.load('../../../.venv/model/grabcut/model_mobilenetv2_exp12_new.pth'))
model.load_state_dict(torch.load('../../../.venv/model/model_mobilenetv2_exp311_f1.pth'))
model = model.to(device)
model.eval()

# Evaluate the model and collect predictions
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)

# Print metrics
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(report)
