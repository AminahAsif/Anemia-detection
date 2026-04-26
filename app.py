import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, render_template, jsonify
import base64
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

RF_MODEL = joblib.load("models/rf_combined.pkl")
SCALER   = joblib.load("models/ml_scaler_combined.pkl")
DL_MODEL = tf.keras.models.load_model("models/mobilenet_final.keras")

def extract_features(img):
    features = []
    img_float = img.astype(np.float32) / 255.0
    for ch in range(3):
        features.append(img_float[:,:,ch].mean())
        features.append(img_float[:,:,ch].std())
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)/255.0
    for ch in range(3):
        features.append(img_hsv[:,:,ch].mean())
        features.append(img_hsv[:,:,ch].std())
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)/255.0
    for ch in range(3):
        features.append(img_lab[:,:,ch].mean())
        features.append(img_lab[:,:,ch].std())
    for ch in range(3):
        hist = cv2.calcHist([img], [ch], None, [32], [0,256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist.tolist())
    return np.array(features, dtype=np.float32)

def preprocess_image(file_bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img   = cv2.resize(img, (224, 224))
    lab   = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l     = clahe.apply(l)
    img_clahe = cv2.merge([l, a, b])
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)
    return img, img_clahe

def get_gradcam(img_rgb):
    img_pre   = preprocess_input(img_rgb.astype(np.float32))
    img_batch = np.expand_dims(img_pre, axis=0)
    last_conv = None
    for layer in reversed(DL_MODEL.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    grad_model = tf.keras.models.Model(
        inputs  = DL_MODEL.inputs,
        outputs = [DL_MODEL.get_layer(last_conv).output,
                   DL_MODEL.output]
    )
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(img_batch)
        loss = pred[:, 0]
    grads    = tape.gradient(loss, conv_out)
    pooled   = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap  = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap  = tf.squeeze(heatmap)
    heatmap  = tf.maximum(heatmap, 0)
    heatmap  = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap  = heatmap.numpy()
    hmap_r   = cv2.resize(heatmap, (224,224))
    hmap_u   = np.uint8(255 * hmap_r)
    hmap_c   = cv2.applyColorMap(hmap_u, cv2.COLORMAP_JET)
    hmap_rgb = cv2.cvtColor(hmap_c, cv2.COLOR_BGR2RGB)
    overlay  = cv2.addWeighted(img_rgb, 0.6, hmap_rgb, 0.4, 0)
    _, buf   = cv2.imencode(".png",
                            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode("utf-8"), float(pred.numpy()[0][0])

def img_to_b64(img_rgb):
    _, buf = cv2.imencode(".png",
                          cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode("utf-8")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file       = request.files["file"]
    mode       = request.form.get("mode", "rf")
    file_bytes = file.read()
    img_rgb, img_clahe = preprocess_image(file_bytes)
    original_b64       = img_to_b64(img_rgb)
    if mode == "rf":
        features = extract_features(img_clahe)
        feat_sc  = SCALER.transform(features.reshape(1,-1))
        proba    = RF_MODEL.predict_proba(feat_sc)[0]
        pred     = int(RF_MODEL.predict(feat_sc)[0])
        conf     = float(proba[pred])
        label    = "Anemic" if pred == 1 else "Normal"
        return jsonify({"label": label,
                        "confidence": round(conf*100, 1),
                        "mode": "Random Forest (78.8% CV Accuracy)",
                        "original": original_b64,
                        "gradcam": None})
    else:
        gradcam_b64, dl_pred = get_gradcam(img_rgb)
        label = "Anemic" if dl_pred > 0.5 else "Normal"
        conf  = dl_pred if dl_pred > 0.5 else 1 - dl_pred
        return jsonify({"label": label,
                        "confidence": round(conf*100, 1),
                        "mode": "MobileNetV2 + Grad-CAM",
                        "original": original_b64,
                        "gradcam": gradcam_b64})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0",
            port=int(os.environ.get("PORT", 5000)))
