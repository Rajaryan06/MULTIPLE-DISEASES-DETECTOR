import matplotlib.pyplot as plt
import numpy as np

# Make sure these variables are already defined from previous steps:
# - model: your trained model
# - X_val: validation images (numpy array)
# - y_val: true validation labels (multi-hot encoded)
# - mlb: your MultiLabelBinarizer (from Step 2)

# 1. Predict on validation data
preds = model.predict(X_val)
pred_labels = (preds > 0.5).astype(int)  # Convert probabilities to 0 or 1

# 2. Decode predictions and true labels to text
true_names = mlb.inverse_transform(y_val)
pred_names = mlb.inverse_transform(pred_labels)

# 3. Select random images to display
num_to_show = 5
indices = np.random.choice(len(X_val), num_to_show, replace=False)

plt.figure(figsize=(15, 8))

for i, idx in enumerate(indices):
    image = X_val[idx]
    true = true_names[idx]
    pred = pred_names[idx]

    plt.subplot(1, num_to_show, i + 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Predicted: {pred}\nActual: {true}', fontsize=10)

plt.tight_layout()
plt.show()