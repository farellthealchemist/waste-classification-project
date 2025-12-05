import os
import cv2

train_path = 'C:/Users/Farel/Desktop/project_sampah/dataset/TRAIN'
categories = ['R', 'O']

print("=== CEK UKURAN FOTO ===")
print()

for category in categories:
    folder_path = os.path.join(train_path, category)
    images = os.listdir(folder_path)
    
    # Ambil 10 foto untuk sample
    sample_images = images[:10]
    
    cat_name = "Recyclable" if category == "R" else "Organic"
    print(f"📂 Kategori: {cat_name}")
    
    for img_name in sample_images:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        
        if img is not None:
            height, width = img.shape[:2]
            print(f"  - {img_name}: {width}x{height} pixels")
    
    print()

print("💡 Kesimpulan: Ukuran foto berbeda-beda")
print("💡 Nanti kita akan resize semua foto jadi 224x224 pixels (ukuran standar untuk AI)")