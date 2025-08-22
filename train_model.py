

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# === CONFIG ===
IMG_SIZE = 224
CSV_PATH = "_annotations.csv"
IMG_FOLDER = "train"

# === LOAD CSV ===
df = pd.read_csv(CSV_PATH)
class_map = {"real": 0, "fake": 1}
df["class_id"] = df["class"].map(class_map)

# === PREPROCESSING ===
def load_image_and_labels(row):
    img_path = os.path.join(IMG_FOLDER, row["filename"])
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0

    x_scale = IMG_SIZE / row["width"]
    y_scale = IMG_SIZE / row["height"]
    xmin = row["xmin"] * x_scale / IMG_SIZE
    ymin = row["ymin"] * y_scale / IMG_SIZE
    xmax = row["xmax"] * x_scale / IMG_SIZE
    ymax = row["ymax"] * y_scale / IMG_SIZE

    bbox = [xmin, ymin, xmax, ymax]
    label = row["class_id"]
    return img, bbox, label

images, bboxes, labels = [], [], []
for _, row in df.iterrows():
    img, bbox, label = load_image_and_labels(row)
    images.append(img)
    bboxes.append(bbox)
    labels.append(label)

X = np.array(images)
y_bbox = np.array(bboxes)
y_class = np.array(labels)

# === MODEL ===
input_layer = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Conv2D(32, (3,3), activation='relu')(input_layer)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)

class_output = layers.Dense(1, activation='sigmoid', name="class_output")(x)
bbox_output = layers.Dense(4, activation='sigmoid', name="bbox_output")(x)

model = models.Model(inputs=input_layer, outputs=[class_output, bbox_output])
model.compile(
    optimizer='adam',
    loss={'class_output': 'binary_crossentropy', 'bbox_output': 'mse'},
    metrics={'class_output': 'accuracy'}
)

# === TRAIN ===
model.fit(
    X, {'class_output': y_class, 'bbox_output': y_bbox},
    epochs=20,
    batch_size=16,
    validation_split=0.2
)

# === PREDICT ===
test_img = X[0:1]
pred_class, pred_bbox = model.predict(test_img)
print("Predicted class:", "fake" if pred_class[0][0] > 0.5 else "real")
print("Predicted bbox:", pred_bbox[0])

# === SAVE MODEL ===
model.save("adidas_detector_model.h5")
