import torch
from PIL import Image
import os
from args import get_args
from utils import resize_box_xyxy
from torchvision.transforms.functional import to_tensor

class ObjDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.args = get_args()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get the row from dataframe
        row = self.df.iloc[idx]

        # 2. Load image
        img_path = row["images"].replace("\\", os.sep)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        # Resize image
        img = img.resize((self.args.image_size, self.args.image_size))
        image = to_tensor(img)

        boxes, labels = [], []
        
        # 3. Load labels
        label_path = row["label_path"].replace("\\", os.sep)
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    cls, xc, yc, bw, bh = map(float, line.split())
                    
                    x1 = (xc - bw/2) * w
                    y1 = (yc - bh/2) * h
                    x2 = (xc + bw/2) * w
                    y2 = (yc + bh/2) * h
                    
                    x1, y1, x2, y2 = resize_box_xyxy(
                        (x1, y1, x2, y2), w, h, 
                        self.args.image_size, self.args.image_size
                    )
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls) + 1)

        # 4. Handle case with no boxes
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

        return image, target