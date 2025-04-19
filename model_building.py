from tensorflow.keras.applications import EfficientNetB0 # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf

# Load EfficientNetB0 without the top layer
base_model = EfficientNetB0(
    include_top=False,
    input_shape=(224, 224, 3),
    weights='imagenet'
)

# Freeze base model layers (optional)
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(6, activation='sigmoid')(x)  # Use len(mlb.classes_) if you have label binarizer

# Final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # For multi-label
    metrics=['accuracy']
)

# Optional: Print summary
model.summary()