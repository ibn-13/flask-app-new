from flask import Flask, request, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import cv2
from sklearn.mixture import GaussianMixture

# --- Konfigurasi dasar ---
app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load model binary classification ---
model = load_model('model_gmm_hasil/resnet_model_gmm_opt_final.keras')
class_labels = ['Caries', 'Non-Caries']
img_size = 224

# --- Fungsi Segmentasi GMM (sesuai code training) ---
def gmm_threshold(img_array):
    img_resized = cv2.resize(img_array, (img_size, img_size))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    pixels = gray.reshape(-1, 1).astype(np.float32)

    gmm = GaussianMixture(
        n_components=2, max_iter=200, tol=1e-2, init_params="kmeans", random_state=42
    )
    gmm.fit(pixels)
    threshold = np.mean(gmm.means_)

    binary_img = np.where(gray > threshold, 255, 0).astype("uint8")
    binary_rgb = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)
    return binary_rgb.astype("float32") / 255.0

# --- Fungsi Preprocessing + Segmentasi ---
def preprocess_and_segment(image_path):
    # Simpan gambar asli
    img_pil = Image.open(image_path).convert('RGB')
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original.jpg')
    img_pil.save(original_path)

    # Konversi ke numpy (RGB)
    img_np = np.array(img_pil)

    # Segmentasi GMM
    segmented = gmm_threshold(img_np)

    # Simpan hasil segmentasi
    segmented_path = os.path.join(app.config['UPLOAD_FOLDER'], 'segmented.jpg')
    seg_disp = (segmented * 255).astype(np.uint8)  # untuk simpan
    Image.fromarray(seg_disp).save(segmented_path)

    # Bentuk array untuk model
    img_array = np.expand_dims(segmented, axis=0)  # (1, 224, 224, 3)
    return img_array, original_path, segmented_path

# --- Fungsi hanya load segmented image ---
def load_segmented_image():
    segmented_path = os.path.join(app.config['UPLOAD_FOLDER'], 'segmented.jpg')
    img = Image.open(segmented_path).convert('RGB')
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Halaman utama ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Upload dan preview gambar ---
@app.route('/preview', methods=['POST'])
def preview():
    file = request.files['image']
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        _, original, segmented = preprocess_and_segment(image_path)

        return render_template('index.html',
                               original_image=original,
                               processed_image=segmented)
    return 'File gambar tidak valid.'

# --- Prediksi gambar ---
@app.route('/predict', methods=['POST'])
def predict():
    img_array = load_segmented_image()
    prediction = model.predict(img_array)[0][0]
    class_idx = 1 if prediction >= 0.5 else 0
    class_name = class_labels[class_idx]
    confidence = prediction if class_idx == 1 else 1 - prediction

    return render_template('index.html',
                           prediction_result=True,
                           class_name=class_name,
                           confidence=round(confidence * 100, 2),
                           original_image='static/original.jpg',
                           processed_image='static/segmented.jpg')

# --- Halaman evaluasi model ---
@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html',
                           cm_path='static/grafik/cm.png',
                           training_plot_path='static/training_plot.png')

# --- Jalankan Flask App ---
if __name__ == '__main__':
    app.run(debug=True)
