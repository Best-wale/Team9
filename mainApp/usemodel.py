import tensorflow as tf
import cv2
import numpy as np

# === CONFIG ===
MODEL_PATH = "C:/Users/USER/Desktop/Projects_Folder/Project_AI_ML/OCR_folder/verification_engine/adidas_detector_model.h5"
IMG_PATH   = "C:/Users/USER/Desktop/Projects_Folder/Project_AI_ML/OCR_folder/verification_engine/test/e47010eca54238fc160c4a9045bf0eef_jpeg.rf.1259bb2175796eceb2ded37f7f0f4c78.jpg"
IMG_SIZE   = 224

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

    # raw confidence score (0.0–1.0)
    confidence_score = float(pred_class[0][0])
    print(confidence_score)
    label = "fake" if confidence_score > 0.5 else "real"
    return label, confidence_score, pred_bbox[0]  # bbox normalized

# === DRAW BOUNDING BOX ===
def draw_bbox(img_path, bbox, label):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    xmin = int(bbox[0] * w)
    ymin = int(bbox[1] * h)
    xmax = int(bbox[2] * w)
    ymax = int(bbox[3] * h)

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.putText(img, label, (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === MAIN EXECUTION ===
if __name__ == "__main__":
    label, score, bbox = predict(IMG_PATH)

    # map to your text labels
    authenticity_text = "Suspicious" if label == "fake" else "Genuine"

    if score > 0.8:
        confidence_text = "High"
    elif score > 0.5:
        confidence_text = "Medium"
    else:
        confidence_text = "Low"

    # your custom analysis message
    analysis_detail = "The logo looks slightly off..."

    # build the JS const block
    analysisText = f"""const analysisText = `
  AUTHENTICITY: {authenticity_text},  
  CONFIDENCE: {confidence_text},  
  ANALYSIS: {analysis_detail}
`;"""

    # output it
    print(analysisText)
    print(f"Bounding Box (normalized): {bbox}")

    # still draw your box if you want
    draw_bbox(IMG_PATH, bbox, authenticity_text)
