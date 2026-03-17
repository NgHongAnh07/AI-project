from args import get_args
import pandas as pd
import os
import torch 
from dataset import ObjDetectionDataset
from torch.utils.data import DataLoader
from model import build_model
from trainer import train_model 

def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def main():
    args = get_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"--> Using device: {device}")

    train_df = pd.read_csv(os.path.join(args.csv_dir, 'train_df.csv'))
    val_df = pd.read_csv(os.path.join(args.csv_dir, 'val_df.csv'))

    train_dataset = ObjDetectionDataset(train_df)
    val_dataset = ObjDetectionDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = build_model(args.backbone, args.num_classes)
    
    print(f"--> Starting training for {args.epochs} epochs...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    print("--> Assignment complete! Check the 'outputs' folder for best_model.pth")

if __name__ == "__main__":
    main()
