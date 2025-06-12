# Automated-quality-assessment-of-crops-using-synthetic-dataset
# Automated Crop Quality Assessment using CNN (PlantVillage Dataset)
# FULL PIPELINE: Synthetic Dataset Generation + Training + Evaluation
# Compatible with Google Colab

# 1. Install and import required libraries
!pip install tensorflow --quiet
!pip install opencv-python --quiet
!pip install tqdm --quiet

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm

# 2. Generate Synthetic PlantVillage-style Dataset
base_path = "/content/plantvillage"
classes = ["Healthy", "Diseased"]
images_per_class = 100
img_height, img_width = 128, 128

for cls in classes:
    os.makedirs(os.path.join(base_path, cls), exist_ok=True)

def generate_leaf_image(is_diseased):
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    img[:, :] = (34, 139, 34)  # Leaf green
    if is_diseased:
        for _ in range(np.random.randint(5, 15)):
            center = tuple(np.random.randint(0, img_height, size=2))
            radius = np.random.randint(5, 15)
            color = (np.random.randint(50, 100), 20, 20)  # dark spot
            cv2.circle(img, center, radius, color, -1)
    return img

for cls in tqdm(classes, desc="Generating synthetic dataset"):
    is_diseased = (cls == "Diseased")
    class_dir = os.path.join(base_path, cls)
    for i in range(images_per_class):
        img = generate_leaf_image(is_diseased)
        img_path = os.path.join(class_dir, f"{cls.lower()}_{i}.jpg")
        Image.fromarray(img).save(img_path)

print(f"Synthetic dataset created at {base_path} with {images_per_class} images per class.")

# 3. Data Preprocessing and Augmentation
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    base_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    base_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

num_classes = len(train_generator.class_indices)

# 4. Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5. Train the Model
epochs = 10  # Increase for better accuracy

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=epochs
)

# 6. Evaluate the Model
loss, acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {acc*100:.2f}%")

# 7. Plot Training History
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()

# 8. Save the Model
model.save("crop_quality_cnn.h5")
print("Model saved as crop_quality_cnn.h5")

# 9. Inference Example (Predict on a new image)
def predict_image(img_path, model, class_indices):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_height, img_width))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    class_names = list(class_indices.keys())
    pred_class = class_names[np.argmax(pred)]
    confidence = np.max(pred)
    return pred_class, confidence

# Example usage (uncomment to test with your own image file):
# pred_class, confidence = predict_image('/content/plantvillage/Healthy/healthy_0.jpg', model, train_generator.class_indices)
# print(f"Predicted Class: {pred_class} (Confidence: {confidence:.2f})")

