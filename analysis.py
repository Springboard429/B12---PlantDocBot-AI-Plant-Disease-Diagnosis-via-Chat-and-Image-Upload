import os
import cv2
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm

print("All libraries imported successfully")

# -------------------------------
# Path to dataset
# -------------------------------
dataset_path = "dataset"

# Get all folders in dataset, ignoring unwanted ones
classes = [cls for cls in os.listdir(dataset_path) 
           if os.path.isdir(os.path.join(dataset_path, cls)) and cls != "PlantVillage"]

print("Classes found:", classes)

# -------------------------------
# Task 2: Plot sample images
# -------------------------------
plt.figure(figsize=(12, 8))
for i, cls in enumerate(classes[:6]):  # Show 6 classes as example
    cls_path = os.path.join(dataset_path, cls)
    
    # Only list image files
    img_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        continue  # Skip if folder has no images
    
    img_name = random.choice(img_files)
    img_path = os.path.join(cls_path, img_name)
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis('off')

plt.tight_layout()
plt.show()

# -------------------------------
# Task 3: Class Distribution & Image Resolution Analysis
# -------------------------------
image_counts = {}
widths, heights, aspect_ratios = [], [], []

for cls in classes:
    cls_path = os.path.join(dataset_path, cls)
    
    # Only image files
    img_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_counts[cls] = len(img_files)
    
    for img_name in img_files:
        img_path = os.path.join(cls_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue  # Skip unreadable files
        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)

# Class distribution table
df_stats = pd.DataFrame({
    "Class": list(image_counts.keys()),
    "Image_Count": list(image_counts.values())
})
print("\nClass Distribution:\n", df_stats)

# Plot class distribution
plt.figure(figsize=(12, 6))
plt.bar(df_stats["Class"], df_stats["Image_Count"], color="skyblue")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Disease/Class")
plt.ylabel("Number of Images")
plt.title("Class Distribution of Plant Disease Dataset")
plt.tight_layout()
plt.show()

# Image resolution statistics
print(f"\nImage Width: min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.2f}")
print(f"Image Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.2f}")
print(f"Aspect Ratio: min={min(aspect_ratios):.2f}, max={max(aspect_ratios):.2f}, mean={np.mean(aspect_ratios):.2f}")
# -------------------------------
# Step 4: Dataset Quality Assessment
# -------------------------------

def detect_blur(image_path, threshold=100):
    img = cv2.imread(image_path)
    if img is None:
        return None, "CORRUPT"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if score < threshold:
        return score, "BLURRY"
    else:
        return score, "CLEAR"

def get_image_hash(image_path):
    try:
        img = Image.open(image_path)
        return imagehash.phash(img)
    except:
        return None

def scan_dataset(dataset_path):
    blurry, clear, corrupt, duplicates = [], [], [], []
    hash_map = {}
    for root, _, files in os.walk(dataset_path):
        for file in tqdm(files, desc="Scanning images"):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(root, file)
                score, status = detect_blur(path)
                if status == "CORRUPT":
                    corrupt.append(path)
                    continue
                elif status == "BLURRY":
                    blurry.append(path)
                else:
                    clear.append(path)
                h = get_image_hash(path)
                if h in hash_map:
                    duplicates.append((hash_map[h], path))
                else:
                    hash_map[h] = path
    return blurry, clear, corrupt, duplicates

# -------------------------------
# Run Step 4 scan & print report
# -------------------------------
blurry, clear, corrupt, duplicates = scan_dataset(dataset_path)

total = len(blurry) + len(clear) + len(corrupt)
print("\nDATASET QUALITY REPORT")
print("-" * 40)
print("Total images   :", total)
print("Clear images   :", len(clear))
print("Blurry images  :", len(blurry))
print("Corrupt images :", len(corrupt))
print("Duplicate pairs:", len(duplicates))
if total > 0:
    print(f"Blurry %       : {(len(blurry)/total)*100:.2f}%")