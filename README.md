# 🧠 Multi-Disease Detector using Deep Learning

A deep learning project for **multi-label classification** of retinal diseases from medical images using **EfficientNetB0**, built with TensorFlow and Keras. This model can detect the presence of multiple diseases in a single image.

---

## 📌 Project Overview

This project uses transfer learning with `EfficientNetB0` to classify retinal images for multiple diseases at once (multi-label classification). It includes preprocessing, data augmentation, model training, evaluation, and predictions.

---

## 🗂️ Project Structure

multi-disease-detector/ │ ├── data/ │ ├── images/ # All retina .jpg images │ └── labels.csv # CSV file with image_id and associated labels │ ├── preprocess_and_save.py # Loads and preprocesses data ├── train_model.py # Model architecture, training, saving ├── evaluate.py # Evaluate & visualize model performance ├── best_model.h5 # Saved model weights (after training) ├





---

## 📸 Dataset Format

- `labels.csv` should look like this:

| image_id   | labels                  |
|------------|--------------------------|
| image_001  | ["Disease A", "Disease C"] |
| image_002  | ["Disease B"]             |

- Images must be named as `<image_id>.jpg` inside `data/images/`.

---

## 🚀 How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt




#requirements.txt

tensorflow
pandas
numpy
opencv-python
scikit-learn
matplotlib


python preprocess_and_save.py

This script:

Loads images

Normalizes and resizes them

Encodes the labels using MultiLabelBinarizer


Train the Model



python train_model.py
This script:

Builds the model using EfficientNetB0

Applies data augmentation

Uses EarlyStopping and ModelCheckpoint

Trains the model and saves best_model.h5

 Evaluate the Model

python evaluate.py
This script:

Evaluates performance on validation set

Plots training history (loss & accuracy)

Generates a classification report

🧠 Model Details
Backbone: EfficientNetB0 (pretrained on ImageNet)

Head: GlobalAveragePooling2D + Dropout + Dense(sigmoid)

Loss Function: binary_crossentropy

Metrics: Accuracy

Output Activation: Sigmoid (for multi-label output)

📊 Example Visualizations
Training/Validation Loss and Accuracy

Per-class classification report

Prediction vs. Ground Truth on sample images

🔮 Future Work
Fine-tune EfficientNetB0 layers

Add web interface using Streamlit

Export to TFLite or ONNX

Handle large-scale image sets with data generators

🧑‍💻 Author

Aryan Raj and Khushi Kumari 


