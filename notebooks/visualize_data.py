import os
import matplotlib.pyplot as plt
import numpy as np

# Path ke dataset
train_path = 'C:/Users/Farel/Desktop/project_sampah/dataset/TRAIN'
test_path = 'C:/Users/Farel/Desktop/project_sampah/dataset/TEST'

print("=" * 60)
print("📊 VISUALISASI DISTRIBUSI DATA")
print("=" * 60)
print()

# Hitung jumlah foto per kategori
categories = ['O', 'R']
category_names = {
    'O': 'Organic',
    'R': 'Recyclable'
}

# Training data
train_counts = {}
for cat in categories:
    folder_path = os.path.join(train_path, cat)
    count = len(os.listdir(folder_path))
    train_counts[cat] = count
    print(f"Training - {category_names[cat]}: {count:,} gambar")

print()

# Testing data
test_counts = {}
for cat in categories:
    folder_path = os.path.join(test_path, cat)
    count = len(os.listdir(folder_path))
    test_counts[cat] = count
    print(f"Testing - {category_names[cat]}: {count:,} gambar")

print()
print(f"Total Training: {sum(train_counts.values()):,} gambar")
print(f"Total Testing: {sum(test_counts.values()):,} gambar")
print()

# Hitung persentase
train_total = sum(train_counts.values())
test_total = sum(test_counts.values())

print("📈 Persentase Distribusi:")
print()
print("Training:")
for cat in categories:
    percentage = (train_counts[cat] / train_total) * 100
    print(f"  - {category_names[cat]}: {percentage:.1f}%")

print()
print("Testing:")
for cat in categories:
    percentage = (test_counts[cat] / test_total) * 100
    print(f"  - {category_names[cat]}: {percentage:.1f}%")

print()

# Visualisasi dengan bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training data
colors = ['#ff9999', '#66b3ff']
axes[0].bar(['Organic', 'Recyclable'], 
            [train_counts['O'], train_counts['R']], 
            color=colors, 
            alpha=0.8,
            edgecolor='black')
axes[0].set_title('Training Data Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Jumlah Gambar')
axes[0].set_ylim(0, max(train_counts.values()) * 1.2)

# Tambah angka di atas bar
for i, (cat, count) in enumerate(train_counts.items()):
    axes[0].text(i, count + 200, f'{count:,}\n({count/train_total*100:.1f}%)', 
                ha='center', fontsize=12, fontweight='bold')

# Testing data
axes[1].bar(['Organic', 'Recyclable'], 
            [test_counts['O'], test_counts['R']], 
            color=colors, 
            alpha=0.8,
            edgecolor='black')
axes[1].set_title('Testing Data Distribution', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Jumlah Gambar')
axes[1].set_ylim(0, max(test_counts.values()) * 1.2)

# Tambah angka di atas bar
for i, (cat, count) in enumerate(test_counts.items()):
    axes[1].text(i, count + 20, f'{count:,}\n({count/test_total*100:.1f}%)', 
                ha='center', fontsize=12, fontweight='bold')

plt.suptitle('Distribusi Dataset - Organic vs Recyclable', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Analisis balance
print()
print("=" * 60)
print("🔍 ANALISIS BALANCE")
print("=" * 60)
print()

ratio = max(train_counts.values()) / min(train_counts.values())
print(f"Rasio Organic:Recyclable = {ratio:.2f}:1")
print()

if ratio < 1.5:
    print("✅ Dataset SANGAT SEIMBANG (rasio < 1.5)")
    print("   AI tidak akan bias ke salah satu kategori")
elif ratio < 2.0:
    print("✅ Dataset CUKUP SEIMBANG (rasio < 2.0)")
    print("   Masih bagus untuk training")
else:
    print("⚠️  Dataset KURANG SEIMBANG (rasio >= 2.0)")
    print("   Pertimbangkan untuk balancing data")

print()
print("💡 Kesimpulan: Dataset siap untuk training!")