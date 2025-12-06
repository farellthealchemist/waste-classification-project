import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Path ke dataset
train_path = 'C:/Users/Farel/Desktop/project_sampah/dataset/TRAIN'
test_path = 'C:/Users/Farel/Desktop/project_sampah/dataset/TEST'

# Parameter
IMG_SIZE = 224
BATCH_SIZE = 32

print("=" * 60)
print("🔄 SETUP DATA GENERATOR")
print("=" * 60)
print()

# Data Augmentation untuk TRAINING
# Ini akan bikin variasi foto (putar, flip, zoom) supaya AI lebih pintar
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalisasi 0-255 jadi 0-1
    rotation_range=20,           # Putar random max 20 derajat
    width_shift_range=0.2,       # Geser horizontal max 20%
    height_shift_range=0.2,      # Geser vertikal max 20%
    shear_range=0.2,             # Shear transformation
    zoom_range=0.2,              # Zoom in/out max 20%
    horizontal_flip=True,        # Flip kiri-kanan
    fill_mode='nearest'          # Cara isi pixel kosong
)

# Untuk TESTING/VALIDATION - TIDAK pakai augmentation
# Hanya normalisasi saja
test_datagen = ImageDataGenerator(rescale=1./255)

print("✅ Training Data Generator:")
print("   - Augmentation: ON (rotation, flip, zoom, shift)")
print("   - Normalization: ON (0-1)")
print()

print("✅ Testing Data Generator:")
print("   - Augmentation: OFF")
print("   - Normalization: ON (0-1)")
print()

# Setup generator untuk TRAINING
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',         # Binary karena cuma 2 kelas (R atau O)
    shuffle=True                 # Acak urutan foto
)

# Setup generator untuk TESTING
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False                # Jangan acak untuk testing
)

print()
print("=" * 60)
print("📊 SUMMARY")
print("=" * 60)
print()

print(f"Training Data:")
print(f"  - Total: {train_generator.samples} gambar")
print(f"  - Classes: {train_generator.class_indices}")
print(f"  - Batches: {len(train_generator)} batches")
print()

print(f"Testing Data:")
print(f"  - Total: {test_generator.samples} gambar")
print(f"  - Classes: {test_generator.class_indices}")
print(f"  - Batches: {len(test_generator)} batches")
print()

# Visualisasi: Tampilkan 1 batch hasil augmentation
print("🖼️  Generating sample batch with augmentation...")
print()

# Ambil 1 batch (32 gambar)
sample_batch, sample_labels = next(train_generator)

# Tampilkan 8 gambar dari batch
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
fig.suptitle('Sample Batch dengan Data Augmentation', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i < len(sample_batch):
        ax.imshow(sample_batch[i])
        label = 'Organic' if sample_labels[i] == 0 else 'Recyclable'
        ax.set_title(f'{label}')
        ax.axis('off')

plt.tight_layout()
plt.show()

print("✅ Data Generator siap digunakan untuk training!")
print()
print("💡 Next: Kita akan pakai generator ini untuk train model CNN")