from args import get_args
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


def plot_learning_curve(train_losses, val_losses, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', marker='o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', marker='s', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, "learning_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def train_model(model, train_loader, val_loader, device):
    args = get_args()
    model = model.to(device)
   
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    patience = args.early_stopping_patience if hasattr(args, 'early_stopping_patience') else 5
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            targets = [
                {
                    'boxes': target['boxes'].to(device=device, dtype=torch.float32),
                    'labels': target['labels'].to(device=device, dtype=torch.int64)
                }
                for target in targets
            ]

            optimizer.zero_grad()

            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(images)

        train_epoch_loss = running_loss / len(train_loader.dataset)

        val_loss = validate_model(model, val_loader, device)
        scheduler.step(val_loss)
        
        train_losses.append(train_epoch_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {train_epoch_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")

        model_to_save = model.module if hasattr(model, 'module') else model
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(model_to_save.state_dict(), 
                      os.path.join(args.out_dir, "best_model.pth"))
            print(f"Saved best model with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        torch.save(checkpoint, os.path.join(args.out_dir, "checkpoint.pth"))

    plot_learning_curve(train_losses, val_losses, args.out_dir)

    model_to_load = model.module if hasattr(model, 'module') else model
    best_model_path = os.path.join(args.out_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model_to_load.load_state_dict(torch.load(best_model_path))
    
    return model


def validate_model(model, val_loader, device):
    model.eval()
    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            targets = [
                {
                    'boxes': target['boxes'].to(device=device, dtype=torch.float32),
                    'labels': target['labels'].to(device=device, dtype=torch.int64)
                }
                for target in targets
            ]

            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())

            val_loss_sum += loss.item() * len(images)
            val_count += len(images)

    val_epoch_loss = val_loss_sum / val_count
    return val_epoch_loss
