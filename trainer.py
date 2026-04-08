import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt # Phải có cái này
from args import get_args

def train_model(model, train_loader, val_loader, device):
    args = get_args()
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

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

        print(f"Epoch {epoch + 1}/{args.epochs} | Train: {train_epoch_loss:.4f} | Val: {val_epoch_loss:.4f}")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))

    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f"Learning Curve - {args.epochs} Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png')
    print("--> SUCCESS: Saved learning_curve.png")
    plt.close()

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

    return val_loss_sum / val_count