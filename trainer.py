import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt 
from args import get_args
from augmentations import Compose, build_train_transforms, build_val_transforms

def train_model(model, train_loader, val_loader, device):
    args = get_args()
    model = model.to(device)
    
    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"--> Starting Training for {args.epochs} Epochs...")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(images)

        train_epoch_loss = running_loss / len(train_loader.dataset)
        val_epoch_loss = validate_model(model, val_loader, device)

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

        print(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            os.makedirs(args.out_dir, exist_ok=True)
            model_path = os.path.join(args.out_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), train_losses, label='Training Loss', color='#1f77b4')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss', color='#d62728', linestyle='--')
    plt.title(f"Hardware Tracking: Learning Curve ({args.epochs} Epochs)", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('learning_curve_final.png', dpi=300)
    plt.close()
    print("--> SUCCESS: Model trained and learning_curve_final.png saved.")

def validate_model(model, val_loader, device):
    val_loss_sum = 0.0
    val_count = 0
    model.train() 

    with torch.no_grad():
        for images, targets in val_loader:
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())

            val_loss_sum += loss.item() * len(images)
            val_count += len(images)

    return val_loss_sum / val_count if val_count > 0 else 0