import os
import pandas as pd

# Update this path if needed
DATASET_PATH = "dataset/skin-ds"

splits = ["train", "val", "test"]

data = []

for split in splits:
    split_path = os.path.join(DATASET_PATH, split)
    classes = os.listdir(split_path)

    for cls in classes:
        cls_path = os.path.join(split_path, cls)
        if os.path.isdir(cls_path):
            num_images = len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            data.append([cls, split, num_images])

df = pd.DataFrame(data, columns=["Class", "Split", "Image_Count"])

# Pivot table for readability
pivot_df = df.pivot(index="Class", columns="Split", values="Image_Count").fillna(0)

print("\n📊 Image Count per Class:\n")
print(pivot_df)

# Save to CSV (for report use)
pivot_df.to_csv("phase1/class_distribution.csv")


import matplotlib.pyplot as plt
import seaborn as sns

# Filter train split
train_df = df[df["Split"] == "train"].sort_values("Image_Count", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(
    x="Class",
    y="Image_Count",
    data=train_df,
    palette="viridis"
)

plt.xticks(rotation=75)
plt.title("Training Set Class Distribution")
plt.xlabel("Lesion Class")
plt.ylabel("Number of Images")

plt.tight_layout()
plt.savefig("phase1/figures/train_class_distribution.png")
plt.show()


import cv2
import random

SAMPLES_PER_CLASS = 4

plt.figure(figsize=(12, 18))
plot_index = 1

for cls in train_df["Class"]:
    cls_path = os.path.join(DATASET_PATH, "train", cls)
    images = os.listdir(cls_path)
    sampled_images = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))

    for img_name in sampled_images:
        img_path = os.path.join(cls_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(len(train_df), SAMPLES_PER_CLASS, plot_index)
        plt.imshow(img)
        plt.axis("off")

        if plot_index % SAMPLES_PER_CLASS == 1:
            plt.ylabel(cls, fontsize=9)

        plot_index += 1

plt.tight_layout()
plt.savefig("phase1/figures/sample_images_per_class.png")
plt.show()
