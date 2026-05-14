import pandas as pd
import os

# Load clinical + functional data
clinical_df = pd.read_csv(
    "data/BC_cardiotox_clinical_and_functional_variables.csv",
    sep=";",
    low_memory=False
)

# Load image labels
image_df = pd.read_csv("data/labels.csv")

# Ensure same number of samples
min_len = min(len(clinical_df), len(image_df))
clinical_df = clinical_df.iloc[:min_len].reset_index(drop=True)
image_df = image_df.iloc[:min_len].reset_index(drop=True)

# Create image_path column
image_dir = "data/images/"
clinical_df["image_path"] = image_df["image_name"].apply(
    lambda x: os.path.join(image_dir, x)
)

# Use image label as target
clinical_df["label"] = image_df["label"]

# Save fused CSV
clinical_df.to_csv("data/fusion_dataset.csv", index=False)

print("✅ Fusion dataset created successfully")
