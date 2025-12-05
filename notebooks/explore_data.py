# Import library yang dibutuhkan
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Path ke dataset (SESUAIKAN dengan lokasi kamu!)
train_path = 'C:\\Users\\Farel\\Desktop\\project_sampah\\dataset\\TRAIN'

# Folder kategori
categories = ['R', 'O']  # R = Recyclable, O = Organic

print("=== CEK ISI DATASET ===")
print()

# Hitung jumlah foto di setiap kategori
for category in categories:
    folder_path = os.path.join(train_path, category)
    jumlah_foto = len(os.listdir(folder_path))
    
    if category == 'R':
        print(f"📦 Recyclable (R): {jumlah_foto} foto")
    else:
        print(f"🌿 Organic (O): {jumlah_foto} foto")

print()
print("Total foto training:", 9999 + 12565)
print("✅ Dataset berhasil ditemukan!")