import torch
import torch.nn as nn
from torchvision import models, transforms
from dataloader import get_dataloader
import os

test_root_path = "maize_dataset_split/test"
test_csv_file_path = "maize_dataset_split/test.csv"
img_size = (224, 224)
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = "weights/best_model.pt"

label_mapping = {
    "Blight": 0,
    "Common_Rust": 1,
    "Gray_Leaf_Spot": 2,
    "Healthy": 3
}

transform_pipeline_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225]),
    ])

test_loader = get_dataloader(
    root_path=test_root_path,
    label_mapping=label_mapping,
    img_size=img_size,
    csv_file_path=test_csv_file_path,
    transforms=transform_pipeline_val,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

def get_resnet18_model(num_classes, pretrained=False):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model = get_resnet18_model(num_classes=4)
model.load_state_dict(torch.load(weights_path)['model_state_dict'])
model = model.to(device)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        pred = torch.argmax(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

accuracy = correct / total * 100
print(f"Test Accuracy: {accuracy:.2f}%")
