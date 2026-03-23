import csv
import os

image_dir = "images"
label_dir = "labels"
output_dir = "CSVs"
output_csv = os.path.join(output_dir, "dataset.csv")

os.makedirs(output_dir, exist_ok=True)

rows = []

for i in range(1, 127):
    base_name = f"img_{i:03}"
    txt_path = os.path.join(label_dir, f"{base_name}.txt")

    img_path = None
    for ext in [".jpg", ".jpeg"]:
        candidate = os.path.join(image_dir, base_name + ext)
        if os.path.exists(candidate):
            img_path = candidate
            break

    if img_path and os.path.exists(txt_path):
        rows.append([img_path, txt_path])
    else:
        print(f"Missing: image or label for {base_name}")

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Images", "Labels"])
    writer.writerows(rows)

print(f" dataset.csv created with {len(rows)} rows")
