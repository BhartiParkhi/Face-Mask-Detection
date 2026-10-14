import kagglehub
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Download Dataset
path = kagglehub.dataset_download("omkargurav/face-mask-dataset")

# Dataset Folder
data_path = os.path.join(path, "data")

images = []
labels = []

categories = ["with_mask", "without_mask"]

for label, category in enumerate(categories):
    folder = os.path.join(data_path, category)

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (100, 100))
        img = img / 255.0

        images.append(img)
        labels.append(label)

# Convert into NumPy Arrays
images = np.array(images, dtype="float32")
labels = np.array(labels)

# Convert Labels
labels = to_categorical(labels, 2)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42
)

print("Training Images :", len(X_train))
print("Testing Images :", len(X_test))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Test Accuracy
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy :", round(accuracy * 100, 2), "%")

# Save Model
model.save("mask_model.h5")

print("Model Saved Successfully!")