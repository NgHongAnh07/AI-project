import torch
import pandas as pd
import os
from args import get_args
from dataset import ObjDetectionDataset
from torch.utils.data import DataLoader
from model import build_model
from trainer import train_model
from augmentations import Compose, build_train_transforms, build_val_transforms

def collate(batch):
    return tuple(zip(*batch))

def main():
    # 0. Configuration
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--> Execution Device: {device}")

    # Initialize professional augmentations
    IMAGE_SIZE = 640 
    train_transform = Compose(build_train_transforms(IMAGE_SIZE))
    val_transform = Compose(build_val_transforms(IMAGE_SIZE))

    # 1. Load Dataframes
    train_path = os.path.join(args.csv_dir, 'train_df.csv')
    val_path = os.path.join(args.csv_dir, 'val_df.csv')
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # 2. Datasets & Loaders
    # FIX: Pass the transforms to the datasets
    train_dataset = ObjDetectionDataset(train_df, transforms=train_transform)
    val_dataset = ObjDetectionDataset(val_df, transforms=val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate, # FIX: Should be 'collate_fn'
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate, # FIX: Should be 'collate_fn'
        num_workers=0
    )

    # 3. Model Initialization
    # num_classes + 1 for background (Standard for Faster R-CNN)
    model = build_model(args.backbone, args.num_classes + 1)
    model.to(device)

    # 4. Training Loop
    print(f"--> Starting training: {args.epochs} epochs")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    print("--> Process completed.")

if __name__ == "__main__":
    main()