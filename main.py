import torch
import pandas as pd
import os
from args import get_args
from dataset import ObjDetectionDataset
from torch.utils.data import DataLoader
from model import build_model
from trainer import train_model

def collate(batch):
    return tuple(zip(*batch))

def main():
    # 0. Configuration
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--> Execution Device: {device}")

    # 1. Load Dataframes
    train_path = os.path.join(args.csv_dir, 'train_df.csv')
    val_path = os.path.join(args.csv_dir, 'val_df.csv')
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # 2. Datasets & Loaders
    train_dataset = ObjDetectionDataset(train_df)
    val_dataset = ObjDetectionDataset(val_df)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate=collate, 
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate=collate, 
        num_workers=0
    )

    # 3. Model Initialization
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