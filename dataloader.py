import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchvision import transforms

class MaizeDiseaseClassification(Dataset):
    def __init__(self, root_path: str, label_mapping: dict, img_size: tuple, csv_file_path: str, transforms=None):
        super().__init__()
        self.root = root_path
        self.label_mapping = label_mapping
        # The tuple should be (target width, target height)
        self.target_img_size = img_size  
        self.data = pd.read_csv(csv_file_path)
        self.columns = list(self.data.columns)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data[self.columns[0]].iloc[idx]
        label_name = self.data[self.columns[1]].iloc[idx]
        label = self.label_mapping[label_name]

        image_path = os.path.join(self.root, label_name, image_name)
        image = Image.open(image_path).convert("RGB") 
        image = image.resize(self.target_img_size, Image.BILINEAR)
        if self.transforms:
            image = self.transforms(image)
        return image, label

from torch.utils.data import DataLoader

def get_dataloader(root_path, label_mapping, img_size, csv_file_path, transforms=None,
                   batch_size=32, shuffle=True, num_workers=4, drop_last=False):
    
    dataset = MaizeDiseaseClassification(
        root_path=root_path,
        label_mapping=label_mapping,
        img_size=img_size,
        csv_file_path=csv_file_path,
        transforms=transforms
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last
    )

    return dataloader

def main():
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225])
    ])
    
    root_path = "maize_dataset_split/train"  
    csv_file_path = "maize_dataset_split/train.csv"  
    img_size = (256, 256)
    label_mapping = {
        "Blight": 0,
        "Common_Rust": 1,
        "Gray_Leaf_Spot": 2,
        "Healthy" : 3
    }

    dataloader = get_dataloader(
        root_path=root_path,
        label_mapping=label_mapping,
        img_size=img_size,
        csv_file_path=csv_file_path,
        transforms=transform_pipeline,
        batch_size=8,
        shuffle=True,
        num_workers=2
    )

    for images, labels in dataloader:
        print(f"Images batch shape: {images.shape}")  
        print(f"Labels batch shape: {labels.shape}")  
        print(f"Labels: {labels}")
        break

if __name__ == "__main__":
    main()        