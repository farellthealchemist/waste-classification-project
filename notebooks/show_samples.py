import os
import cv2
import matplotlib.pyplot as plt
import random

# Path ke dataset
train_path = 'C:/Users/Farel/Desktop/project_sampah/dataset/TRAIN'

print("🔄 Memuat foto...")

# Fungsi untuk tampilkan foto
def show_sample_images():
    categories = {'R': 'Recyclable', 'O': 'Organic'}
    
    # Bikin figure untuk tampilkan 10 foto (5 R, 5 O)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Contoh Foto dari Dataset', fontsize=16)
    
    for row, (cat_code, cat_name) in enumerate(categories.items()):
        folder_path = os.path.join(train_path, cat_code)
        all_images = os.listdir(folder_path)
        
        print(f"📂 Mengambil 5 foto dari kategori {cat_name}...")
        
        # Ambil 5 foto random
        sample_images = random.sample(all_images, 5)
        
        for col, img_name in enumerate(sample_images):
            img_path = os.path.join(folder_path, img_name)
            
            try:
                # Baca gambar
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"⚠️ Gagal baca: {img_name}")
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR ke RGB
                
                # Tampilkan
                axes[row, col].imshow(img)
                axes[row, col].set_title(f'{cat_name}')
                axes[row, col].axis('off')
                
            except Exception as e:
                print(f"❌ Error pada {img_name}: {e}")
    
    plt.tight_layout()
    
    print("\n✅ Foto siap ditampilkan!")
    print("💡 Tutup window untuk melanjutkan...")
    
    plt.show()  # Ini akan buka window
    
    print("✅ Selesai!")

# Jalankan fungsi
try:
    show_sample_images()
except Exception as e:
    print(f"❌ Error: {e}")