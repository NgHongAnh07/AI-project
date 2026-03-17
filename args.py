import argparse

def get_args():
    parser = argparse.ArgumentParser(description="AI Training Pipeline for Object Detection")    
    parser.add_argument('--csv_dir', type=str, default='data/CSVs', 
                        help='Directory containing train_df.csv and val_df.csv')
    parser.add_argument('--out_dir', type=str, default='outputs', 
                        help='Directory to save the best model weights')
    
    parser.add_argument('--backbone', type=str, default='fasterrcnn_resnet50_fpn', 
                        help='Backbone architecture (resnet50 or mobilenet)')
    parser.add_argument('--num_classes', type=int, default=2, 
                        help='Number of classes (including background)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (smaller is safer for Faster R-CNN)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.0005, help='Weight decay for the optimizer')
    
    return parser.parse_args()