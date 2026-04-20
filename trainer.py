from args import get_args
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import show_batch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model(model,train_loader,val_loader,device):
    args = get_args()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.wd)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images,targets in train_loader:
            images = [image.to(device=device,dtype=torch.float32) for image in images]
            targets = [
                {
                'boxes':target['boxes'].to(device=device,dtype=torch.float32),
                'labels':target['labels'].to(device=device,dtype=torch.int64)
                }
                for target in targets
            ]
            #print("length of images:",len(images,))
            
            #show_batch(images, targets)
            optimizer.zero_grad()
            
            loss_dict = model(images,targets)
            loss = sum(loss_value for loss_value in loss_dict.values())

            loss.backward()
            optimizer.step()

            running_loss += loss.item() *len(images)

        train_epoch_loss = running_loss/ len(train_loader.dataset)

        val_loss = validate_model(model,val_loader,device)

        scheduler.step(val_loss)

        train_losses.append(train_epoch_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_epoch_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))

    plot_learning_curve(train_losses, val_losses, args.out_dir)
        

def validate_model(model, val_loader, device):
    model.train()

    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            targets = [
                {
                    "boxes": target["boxes"].to(device=device, dtype=torch.float32),
                    "labels": target["labels"].to(device=device, dtype=torch.int64)
                }
                for target in targets
            ]

          
            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())
            

            val_loss_sum += loss.item() * len(images)
            val_count += len(images)

    val_epoch_loss = val_loss_sum / val_count
    return val_epoch_loss

def plot_learning_curve(train_losses, val_losses, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, marker='o', label='Training Loss')
    plt.plot(epochs, val_losses, marker='o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(out_dir, "learning_curve.png")
    plt.savefig(save_path)
    plt.show()
    plt.close()
