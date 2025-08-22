import tensorflow as tf
import cv2
import numpy as np

# === CONFIG ===
MODEL_PATH = "adidas_detector_model.h5"
IMG_PATH = "C:/Users/USER/Desktop/Projects_Folder/Project_AI_ML/OCR_folder/verification_engine/test/e47010eca54238fc160c4a9045bf0eef_jpeg.rf.1259bb2175796eceb2ded37f7f0f4c78.jpg"
IMG_SIZE = 224

# === LOAD MODEL ===
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# === PREPROCESS IMAGE ===
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    return np.expand_dims(img, axis=0)

# === RUN INFERENCE ===
def predict(img_path):
    input_tensor = preprocess_image(img_path)
    pred_class, pred_bbox = model.predict(input_tensor)

    label = "fake" if pred_class[0][0] > 0.5 else "real"
    bbox = pred_bbox[0]  # [xmin, ymin, xmax, ymax] normalized
    return label, bbox

# === DRAW BOUNDING BOX ===
def draw_bbox(img_path, bbox, label):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    xmin = int(bbox[0] * w)
    ymin = int(bbox[1] * h)
    xmax = int(bbox[2] * w)
    ymax = int(bbox[3] * h)

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === MAIN EXECUTION ===
if __name__ == "__main__":
    label, bbox = predict(IMG_PATH)
    print(f"Prediction: {label}")
    print(f"Bounding Box (normalized): {bbox}")
    draw_bbox(IMG_PATH, bbox, label)
