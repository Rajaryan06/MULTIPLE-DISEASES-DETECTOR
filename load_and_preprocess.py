import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import MultiLabelBinarizer

# Load CSV file with image IDs and labels
df = pd.read_csv("data/labels.csv")  # Make sure this path is correct
df['labels'] = df['labels'].apply(eval)  # Convert string list to actual list

# Encode labels (Multi-label binarization)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['labels'])

# Directory where images are stored
IMAGE_DIR = "data/images"  # Make sure this folder contains your retina images

# Function to load and resize images
def load_images(image_ids, target_size=(224, 224)):
    images = []
    for image_id in image_ids:
        path = os.path.join(IMAGE_DIR, image_id + ".jpg")
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: {path} not found or can't be read")
            continue
        img = cv2.resize(img, target_size)
        img = img / 255.0  # Normalize
        images.append(img)
    return np.array(images)

# Load images
X = load_images(df['image_id'].values)