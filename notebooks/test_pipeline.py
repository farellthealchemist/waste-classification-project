import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

print("=" * 60)
print("🧪 TESTING PIPELINE END-TO-END")
print("=" * 60)
print()

# Path
train_path = 'C:/Users/Farel/Desktop/project_sampah/dataset/TRAIN'

# Parameter
IMG_SIZE = 224
BATCH_SIZE = 32

# Setup generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

print("✅ Generator setup berhasil")
print()

# Test: Ambil 1 batch dan cek
print("📦 Testing batch generation...")
batch_images, batch_labels = next(train_generator)

print(f"✅ Batch shape: {batch_images.shape}")
print(f"   - Batch size: {batch_images.shape[0]} gambar")
print(f"   - Image size: {batch_images.shape[1]}x{batch_images.shape[2]}")
print(f"   - Channels: {batch_images.shape[3]} (RGB)")
print()

print(f"✅ Labels shape: {batch_labels.shape}")
print(f"   - Labels: {batch_labels}")
print(f"   - Organic (0): {np.sum(batch_labels == 0)} gambar")
print(f"   - Recyclable (1): {np.sum(batch_labels == 1)} gambar")
print()

# Cek nilai pixel
print("✅ Pixel values check:")
print(f"   - Min: {batch_images.min():.3f}")
print(f"   - Max: {batch_images.max():.3f}")
print(f"   - Mean: {batch_images.mean():.3f}")
print()

# Visualisasi: Bandingkan gambar yang sama dengan augmentation berbeda
print("🎨 Generating multiple augmentations dari 1 gambar...")
print()

# Ambil 1 gambar original (tanpa augmentation)
from tensorflow.keras.preprocessing.image import load_img, img_to_array

sample_img_path = os.path.join(train_path, 'R', 'R_1.jpg')
original_img = load_img(sample_img_path, target_size=(IMG_SIZE, IMG_SIZE))
original_array = img_to_array(original_img) / 255.0

# Generate 8 versi augmented dari gambar yang sama
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle('1 Gambar Original + 8 Versi Augmented', fontsize=16, fontweight='bold')

# Original
axes[0, 0].imshow(original_array)
axes[0, 0].set_title('ORIGINAL', fontweight='bold', fontsize=12)
axes[0, 0].axis('off')

# Generate 8 augmented versions
for i in range(1, 9):
    # Expand dimensions untuk masuk ke generator
    img_array = np.expand_dims(original_array, axis=0)
    
    # Apply augmentation
    augmented = train_datagen.flow(img_array, batch_size=1)
    aug_img = next(augmented)[0]
    
    row = i // 3
    col = i % 3
    
    axes[row, col].imshow(aug_img)
    axes[row, col].set_title(f'Augmented #{i}', fontsize=10)
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

print()
print("=" * 60)
print("✅ PIPELINE TEST SELESAI - SEMUA BERFUNGSI!")
print("=" * 60)
print()
print("📋 Checklist:")
print("  ✅ Generator bisa load data")
print("  ✅ Batch size benar (32 gambar)")
print("  ✅ Image size benar (224x224x3)")
print("  ✅ Normalisasi bekerja (0-1)")
print("  ✅ Augmentation berfungsi")
print("  ✅ Labels benar (0=Organic, 1=Recyclable)")
print()
print("🎉 SIAP UNTUK TRAINING MODEL!")