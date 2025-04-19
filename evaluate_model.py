import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# 1. Evaluate on validation data
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# 2. Plot loss and accuracy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy over Epochs')
plt.show()

# 3. Predictions
preds = model.predict(X_val)
pred_labels = (preds > 0.5).astype(int)

# 4. Classification report
print(classification_report(y_val, pred_labels, target_names=mlb.classes_))