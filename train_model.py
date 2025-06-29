import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from io import BytesIO
import base64
import plotly.express as px # Import plotly express
import plotly.graph_objects as go # Import plotly graph objects
import json # Import json
import plotly.utils # Import plotly.utils
import albumentations as A  # Untuk augmentasi data
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support

print("1. Memulai preprocessing dataset...")

def augment_image(img):
    """Fungsi untuk augmentasi gambar"""
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(rotate=45, scale=0.1, translate_percent=0.0625, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.3),
    ])
    return transform(image=img)['image']

def load_images_from_folder(folder, label, img_size=(128, 128), augment=True):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            # Konversi ke RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalisasi gambar
            img = img.astype(np.float32) / 255.0
            
            # Resize dengan interpolasi yang lebih baik
            img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
            
            # Tambahkan gambar asli
            images.append(img_resized.flatten())
            labels.append(label)
            
            # Augmentasi jika diaktifkan
            if augment:
                augmented_img = augment_image(img_resized)
                images.append(augmented_img.flatten())
                labels.append(label)
        else:
            print(f"[WARNING] Gagal membaca gambar: {img_path}")
    return images, labels

# Path dataset lokal
base_path = 'dataset/garbage classification'  # Sesuaikan path sesuai struktur kamu
class_mapping = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash"
}

# Mapping hasil prediksi untuk label grafik
class_names = {
    0: "Karton",
    1: "Kaca",
    2: "Kaleng",
    3: "Kertas",
    4: "Plastik",
    5: "Sampah Lainnya"
}

X, y = [], []

# Loop semua folder
for label, folder_name in class_mapping.items():
    folder_path = os.path.join(base_path, folder_name)
    if not os.path.exists(folder_path):
        print(f"[PERINGATAN] Folder tidak ditemukan: {folder_path}")
        continue
    print(f"Memuat gambar dari folder: {folder_name}")
    images, labels = load_images_from_folder(folder_path, label, augment=True)
    X.extend(images)
    y.extend(labels)

# Tambahkan penanganan jika tidak ada data dimuat
if not X:
    raise ValueError("Tidak ada data yang dimuat. Pastikan dataset tersedia dan formatnya benar.")

X = np.array(X)
y = np.array(y)

print(f"\nJumlah data total: {len(X)}")

# Debugging: Tampilkan shape data setelah loading
print(f"[DEBUG] Shape X setelah loading: {X.shape}")
print(f"[DEBUG] Shape y setelah loading: {y.shape}")

# Validasi dimensi array (kembalikan karena penting untuk scaling dan model)
if X.ndim == 1:
    print("[INFO] Data berupa array 1D → reshape ke 2D...")
    X = X.reshape(len(X), -1)
elif X.ndim > 2:
    print("[INFO] Data memiliki lebih dari 2 dimensi → reshape ulang...")
    X = X.reshape(len(X), -1)

print(f"Shape X setelah reshape: {X.shape}")

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\n2. Mulai pelatihan model...")

# Latih model dengan parameter yang dioptimalkan
model = RandomForestClassifier(
    n_estimators=500,          # Meningkatkan jumlah pohon
    max_depth=20,              # Meningkatkan kedalaman maksimum
    min_samples_split=2,       # Mengurangi minimum samples untuk split
    min_samples_leaf=1,        # Mengurangi minimum samples per leaf
    max_features='sqrt',       # Menggunakan sqrt dari jumlah fitur
    bootstrap=True,            # Menggunakan bootstrap sampling
    random_state=42,
    n_jobs=-1,                 # Menggunakan semua core CPU
    class_weight='balanced'    # Menangani ketidakseimbangan kelas
)

# Tambahkan validasi data sebelum training
print("\nValidasi data sebelum training:")
print(f"Jumlah sampel per kelas:")
for label in np.unique(y):
    count = np.sum(y == label)
    print(f"{class_names[label]}: {count} sampel")

# Tambahkan cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Rata-rata CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Latih model
model.fit(X_train, y_train)

# Evaluasi lebih detail
from sklearn.metrics import precision_recall_fscore_support

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi per kelas
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

print("\nEvaluasi per kelas:")
for i, class_name in class_names.items():
    print(f"\n{class_name}:")
    print(f"Precision: {precision[i]:.3f}")
    print(f"Recall: {recall[i]:.3f}")
    print(f"F1-score: {f1[i]:.3f}")
    print(f"Support: {support[i]}")

# Evaluasi keseluruhan
acc = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Model: {acc * 100:.2f}%")
report = classification_report(y_test, y_pred, target_names=[class_names[i] for i in sorted(class_names.keys())])
print("\nLaporan Klasifikasi:\n", report)

# Confusion Matrix (Plotly)
cm = confusion_matrix(y_test, y_pred)
cm_labels = [class_names[i] for i in sorted(class_names.keys())]
fig_cm = go.Figure(data=go.Heatmap(
                   z=cm,
                   x=cm_labels,
                   y=cm_labels,
                   colorscale='Blues'))
fig_cm.update_layout(
    title='Confusion Matrix - Klasifikasi Jenis Sampah',
    xaxis_title='Prediksi',
    yaxis_title='Aktual'
)

# Feature Importance (Matplotlib - sementara kembali ke gambar statis)
feature_importances = model.feature_importances_
top_idx = np.argsort(feature_importances)[-10:]  # Ambil 10 pixel terpenting

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances[top_idx], y=np.arange(len(top_idx)), orient='h') # Perbaiki y-axis
plt.yticks(ticks=np.arange(len(top_idx)), labels=[str(i) for i in top_idx])
plt.xlabel("Tingkat Kepentingan")
plt.ylabel("Indeks Pixel")
plt.title("Top 10 Pixel Terpenting dalam Prediksi")
plt.tight_layout()

buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
feature_importance_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
buffer.close()
plt.close()

# Siapkan data untuk disimpan ke JSON
analysis_results = {
    'confusion_matrix_graph': json.dumps(fig_cm, cls=plotly.utils.PlotlyJSONEncoder),
    'feature_importance_image': f"data:image/png;base64,{feature_importance_base64}",
    'accuracy': acc,
    'classification_report': report
}

# Simpan hasil analisis ke file JSON
with open('analysis_results.json', 'w') as f:
    json.dump(analysis_results, f, indent=4)

# Simpan model
joblib.dump(model, 'model.pkl')

# Simpan scaler
joblib.dump(scaler, 'scaler.pkl')

print("\nModel, scaler, grafik analisis (Confusion Matrix Plotly, Feature Importance Matplotlib), dan hasil analisis berhasil diproses dan disimpan.")