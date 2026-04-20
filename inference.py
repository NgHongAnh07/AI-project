import torch
from PIL import Image, ImageDraw
import os
from model import build_model 
from args import get_args
from torchvision.transforms.functional import to_tensor

def run_inference():
    """
    Perform object detection inference on unseen images using the trained Faster R-CNN model.
    """
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(args.backbone, args.num_classes + 1)
    
    # Check for the best weights
    model_path = os.path.join(args.out_dir, 'best_model_30_epochs.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(args.out_dir, 'best_model.pth')

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    print(f"--> Loading model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_folder = "test_images" 
    output_folder = "test_results"
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(test_folder):
        print(f"Error: Folder '{test_folder}' does not exist.")
        return

    for img_name in os.listdir(test_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_folder, img_name)
            original_img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = original_img.size
            
            input_size = 640 
            img_resized = original_img.resize((input_size, input_size)) 
            img_tensor = to_tensor(img_resized).unsqueeze(0).to(device)

            with torch.no_grad():
                predictions = model(img_tensor)

            draw = ImageDraw.Draw(original_img)
            pred_boxes = predictions[0]['boxes'].detach().cpu().numpy()
            pred_scores = predictions[0]['scores'].detach().cpu().numpy()

            for box, score in zip(pred_boxes, pred_scores):
                if score > 0.5:
                    xmin, ymin, xmax, ymax = box
                    xmin = (xmin / input_size) * orig_w
                    ymin = (ymin / input_size) * orig_h
                    xmax = (xmax / input_size) * orig_w
                    ymax = (ymax / input_size) * orig_h

                    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=5)
                    draw.text((xmin, ymin - 15), f"Laptop: {score:.2f}", fill="red")

            save_path = os.path.join(output_folder, f"result_{img_name}")
            original_img.save(save_path)
            print(f"Generated: {save_path}")

if __name__ == "__main__":
    run_inference()
