import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Path ke dataset
train_path = 'C:/Users/Farel/Desktop/project_sampah/dataset/TRAIN'
test_path = 'C:/Users/Farel/Desktop/project_sampah/dataset/TEST'

# Parameter
IMG_SIZE = 224  # Ukuran standar untuk CNN
BATCH_SIZE = 32  # Jumlah foto yang diproses sekaligus

print("=" * 50)
print("📂 SETUP DATA LOADING")
print("=" * 50)
print()

# Kategori
categories = ['R', 'O']
category_names = {
    'R': 'Recyclable',
    'O': 'Organic'
}

print("📊 Info Dataset:")
print(f"  - Ukuran gambar: {IMG_SIZE}x{IMG_SIZE} pixels")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Kategori: {len(categories)} (Recyclable, Organic)")
print()

# Fungsi untuk load dan preprocess satu gambar
def load_and_preprocess_image(img_path):
    """
    Load gambar, resize, dan normalisasi
    """
    # Baca gambar
    img = cv2.imread(img_path)
    
    # Resize ke 224x224
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Convert BGR ke RGB (OpenCV pakai BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalisasi: ubah dari 0-255 jadi 0-1
    img = img / 255.0
    
    return img

# Test function dengan 1 gambar
print("🧪 Testing load & preprocess function...")
test_img_path = os.path.join(train_path, 'R', 'R_1.jpg')
test_img = load_and_preprocess_image(test_img_path)

print(f"✅ Berhasil!")
print(f"  - Shape gambar: {test_img.shape}")
print(f"  - Min pixel value: {test_img.min():.3f}")
print(f"  - Max pixel value: {test_img.max():.3f}")
print()

# Visualisasi before & after preprocessing
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Before (original)
img_original = cv2.imread(test_img_path)
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
axes[0].imshow(img_original)
axes[0].set_title(f'Before\nUkuran: {img_original.shape[1]}x{img_original.shape[0]}')
axes[0].axis('off')

# After (preprocessed)
axes[1].imshow(test_img)
axes[1].set_title(f'After\nUkuran: {IMG_SIZE}x{IMG_SIZE}, Normalized')
axes[1].axis('off')

plt.suptitle('Preprocessing: Resize & Normalization', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("✅ Preprocessing function siap digunakan!")