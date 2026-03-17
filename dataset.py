import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class ObjDetectionDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms if transforms else T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        label_path = self.df.iloc[idx, 1]

        
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) == 5:

                        labels.append(int(data[0]))
                        boxes.append([float(x) for x in data[1:]])

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        
        if self.transforms:
            img = self.transforms(img)

        return img, target