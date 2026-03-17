import os
import pandas as pd
from sklearn.model_selection import train_test_split

image_dir = "data/images"
label_dir = "data/labels"
csv_dir = "data/CSVs"
os.makedirs(csv_dir, exist_ok=True)

valid_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG')
images = sorted([
    f for f in os.listdir(image_dir) 
    if f.lower().endswith(valid_extensions)
])

data_rows = []
for img in images:
    image_path = os.path.join(image_dir, img)

    label_name = os.path.splitext(img)[0] + ".txt"
    label_path = os.path.join(label_dir, label_name)
    
    if os.path.exists(label_path):
        data_rows.append([image_path, label_path])
    else:
        print(f"⚠️ Warning: Missing label for {img}")

df = pd.DataFrame(data_rows, columns=["images", "labels"])

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv(os.path.join(csv_dir, "train_df.csv"), index=False)
val_df.to_csv(os.path.join(csv_dir, "val_df.csv"), index=False)

print(f"Success!")
print(f"- Total images found: {len(df)}")
print(f"- Saved {len(train_df)} samples to train_df.csv")
print(f"- Saved {len(val_df)} samples to val_df.csv")