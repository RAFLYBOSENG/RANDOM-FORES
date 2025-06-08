from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
import plotly.express as px
import json
import joblib
import plotly.utils

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Muat model
try:
    model = joblib.load('model.pkl')
except Exception as e:
    raise FileNotFoundError("File model.pkl tidak ditemukan. Jalankan train_model.py terlebih dahulu.")

# Mapping hasil prediksi
class_names = {
    0: "Karton",
    1: "Kaca",
    2: "Kaleng",
    3: "Kertas",
    4: "Plastik",
    5: "Sampah Lainnya"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocessing gambar
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konversi BGR ke RGB
            img = cv2.resize(img, (64, 64))
            img_flattened = img.flatten().reshape(1, -1) # Reshape untuk prediksi tunggal

            # Muat scaler
            try:
                scaler = joblib.load('scaler.pkl')
                img_flattened = scaler.transform(img_flattened)
            except:
                print("[WARNING] File scaler.pkl tidak ditemukan. Prediksi mungkin tidak akurat.")

            # Lakukan prediksi dan dapatkan probabilitas
            prediction = model.predict(img_flattened)
            predicted_class = class_names[prediction[0]]

            # Dapatkan probabilitas prediksi
            probabilities = model.predict_proba(img_flattened)[0]
            # Format probabilitas untuk ditampilkan
            formatted_probabilities = {
                class_names[i]: prob * 100 for i, prob in enumerate(probabilities)
            }

            return render_template("result.html", image=filename, result=predicted_class, probabilities=formatted_probabilities)
    return render_template("index.html")

@app.route("/analysis")
def analysis():
    # Ambil data distribusi kelas otomatis dari folder dataset
    base_path = 'dataset/garbage classification'
    class_mapping = {
        0: "cardboard",
        1: "glass",
        2: "metal",
        3: "paper",
        4: "plastic",
        5: "trash"
    }

    class_counts = {}
    for label, folder_name in class_mapping.items():
        folder_path = os.path.join(base_path, folder_name)
        count = len(os.listdir(folder_path)) if os.path.exists(folder_path) else 0
        class_counts[class_names[label]] = count

    # Buat grafik distribusi menggunakan Plotly
    fig_dist = px.bar(
        x=list(class_counts.keys()),
        y=list(class_counts.values()),
        labels={'x': 'Kategori', 'y': 'Jumlah Gambar'},
        title='Distribusi Jumlah Gambar per Kelas',
        color_discrete_sequence=['#17949b']
    )
    graphJSON_distribution = json.dumps(fig_dist, cls=plotly.utils.PlotlyJSONEncoder)

    # Muat hasil analisis dari file JSON
    analysis_data = {}
    try:
        with open('analysis_results.json', 'r') as f:
            analysis_data = json.load(f)
    except FileNotFoundError:
        print("[WARNING] File analysis_results.json tidak ditemukan. Jalankan train_model.py terlebih dahulu.")
    except Exception as e:
        print(f"[ERROR] Gagal memuat analysis_results.json: {e}")

    # Ambil data grafik dan metrik dari data analisis
    confusion_matrix_graph = analysis_data.get('confusion_matrix_graph', None)
    feature_importance_image = analysis_data.get('feature_importance_image', '')
    accuracy = analysis_data.get('accuracy', None)
    classification_report_text = analysis_data.get('classification_report', '')

    return render_template(
        "analysis.html",
        distribution_graph=graphJSON_distribution,
        confusion_matrix_graph=confusion_matrix_graph,
        feature_importance_image=feature_importance_image,
        accuracy=accuracy,
        classification_report_text=classification_report_text
    )

if __name__ == "__main__":
    app.run(debug=True)