from sklearn.model_selection import train_test_split
import numpy as np

# Load your data from step 2
# Assuming you've already run step2_load_and_preprocess.py
# And you have variables X (images), y (labels)

# If not, you can load them from a .npy file (optional if saved earlier)
# X = np.load('X.npy')
# y = np.load('y.npy')

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save the splits (optional)
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)

print("Data split completed:")
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")