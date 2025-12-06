import os
import time
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

print("=" * 60)
print("⚡ PERFORMANCE & MEMORY TEST")
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

print(f"📊 Dataset Info:")
print(f"   - Total images: {train_generator.samples:,}")
print(f"   - Total batches: {len(train_generator):,}")
print(f"   - Images per batch: {BATCH_SIZE}")
print()

# Test: Load 10 batches dan ukur waktu
print("🧪 Test: Loading 10 batches...")
print()

times = []
for i in range(10):
    start_time = time.time()
    batch_images, batch_labels = next(train_generator)
    end_time = time.time()
    
    elapsed = end_time - start_time
    times.append(elapsed)
    
    print(f"   Batch {i+1}: {elapsed:.3f} seconds | Shape: {batch_images.shape} | Labels: O={np.sum(batch_labels==0)}, R={np.sum(batch_labels==1)}")

print()
avg_time = np.mean(times)
print(f"⏱️  Average time per batch: {avg_time:.3f} seconds")
print(f"📈 Estimated time for 1 epoch (706 batches): {(avg_time * 706 / 60):.1f} minutes")
print()

# Memory usage estimation
memory_per_batch = (BATCH_SIZE * IMG_SIZE * IMG_SIZE * 3 * 4) / (1024**2)  # MB
print(f"💾 Memory usage:")
print(f"   - Per batch: ~{memory_per_batch:.1f} MB")
print(f"   - Per epoch: ~{memory_per_batch * 706 / 1024:.2f} GB")
print()

# Visualisasi loading time
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), times, marker='o', linewidth=2, markersize=8, color='#3498db')
plt.axhline(y=avg_time, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_time:.3f}s')
plt.xlabel('Batch Number', fontsize=12)
plt.ylabel('Loading Time (seconds)', fontsize=12)
plt.title('Batch Loading Performance Test', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print()
print("=" * 60)
print("✅ PERFORMANCE TEST SELESAI")
print("=" * 60)
print()

# Analisis
if avg_time < 0.5:
    print("🚀 EXCELLENT: Loading sangat cepat!")
elif avg_time < 1.0:
    print("✅ BAGUS: Loading speed normal")
else:
    print("⚠️  WARNING: Loading agak lambat, tapi masih OK")

print()
print("💡 Tips untuk speed up training nanti:")
print("   1. Tutup aplikasi lain yang berat")
print("   2. Pastikan laptop dalam keadaan charging")
print("   3. Set laptop ke 'High Performance' mode")
print("   4. Jangan membuka banyak tab browser")