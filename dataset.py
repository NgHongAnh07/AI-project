import torch
from PIL import Image
import os
from args import get_args

from torchvision.transforms.functional import to_tensor

class ObjDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None):
        """
        Args:
            df: Dataframe containing image and label paths
            transforms: Professional augmentation pipeline from augmentations.py
        """
        self.df = df.reset_index(drop=True)
        self.args = get_args()
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get the row from dataframe
        row = self.df.iloc[idx]

        # 2. Load image
        img_path = row["images"].replace("\\", os.sep) 
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        boxes, labels = [], []
        
        # 3. Load labels (YOLO format: xc, yc, bw, bh normalized)
        label_path = row["labels"].replace("\\", os.sep) 
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    cls, xc, yc, bw, bh = map(float, line.split())
                    
                    # Convert YOLO format to Pascal VOC (x1, y1, x2, y2) in absolute pixels
                    x1 = (xc - bw/2) * w
                    y1 = (yc - bh/2) * h
                    x2 = (xc + bw/2) * w
                    y2 = (yc + bh/2) * h
                    
                    boxes.append([x1, y1, x2, y2])
                    # Class 0 becomes Class 1 (Class 0 is reserved for background)
                    labels.append(int(cls) + 1)

        # 4. Prepare target dictionary
        if len(boxes) == 0:
            target_boxes = torch.zeros((0, 4), dtype=torch.float32)
            target_labels = torch.zeros((0,), dtype=torch.int64)
        else:
            target_boxes = torch.tensor(boxes, dtype=torch.float32)
            target_labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": target_boxes,
            "labels": target_labels,
            "image_id": torch.tensor([idx]),
        }

        # 5. Apply Professional Augmentations
        if self.transforms is not None:
            image, target = self.transforms(img, target)
        else:
            # Fallback to basic tensor conversion if no transforms are provided
            image = to_tensor(img)

        return image, target