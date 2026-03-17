import os
import csv


image_dir = "data/images"
label_dir = "data/labels"
csv_dir = "data/CSVs"
csv_path = os.path.join(csv_dir, "dataset.csv")

os.makedirs(csv_dir, exist_ok=True)

images = sorted(os.listdir(image_dir))

with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["images", "labels"]) 

    for img in images:
        image_path = os.path.join(image_dir, img)
        label_name = os.path.splitext(img)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)
        writer.writerow([image_path, label_path])

print("dataset.csv has been created successfully!")