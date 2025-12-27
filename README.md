# 🗑️ Sistem Klasifikasi Sampah Pintar

Aplikasi pintar yang bisa **mengenali jenis sampah** hanya dengan foto! Menggunakan teknologi **Kecerdasan Buatan (AI)** untuk membantu memilah sampah dengan benar.

---

## 🌟 Ini Aplikasi apa?

Aplikasi ini adalah **"otak digital"** yang sudah dilatih untuk membedakan 2 jenis sampah:
- 🌿 **Sampah Organik** - Sisa makanan, kulit buah, daun
- ♻️ **Sampah Daur Ulang** - Plastik, kertas, kaleng, botol

**Cara kerjanya sederhana:**
1. Upload foto sampah
2. AI akan menganalisis dalam hitungan detik
3. Dapat hasil: jenis sampah + tingkat keyakinan AI

---

## 🎯 Seberapa Akurat?

✅ **Tingkat Akurasi: 89.91%** (hampir 90 dari 100 prediksi benar!)

Artinya: Dari 100 foto sampah, aplikasi ini bisa menebak dengan benar sekitar 90 kali. Itu angka yang sangat bagus untuk AI!

---

## 🌐 Coba Langsung!

**Aplikasi sudah online dan bisa dipakai gratis:**
👉 [Buka Aplikasi](https://waste-classification-project-pwaiwcwtb5ubhw3smvpsh6.streamlit.app)

Tidak perlu install apa-apa, langsung buka di browser!

---

## 📊 Data & Pelatihan

**Berapa banyak "latihan" yang dilakukan AI?**
- 📚 Dilatih dengan **22,564 foto sampah** (seperti siswa belajar dari 22 ribu contoh soal!)
- ✅ Diuji dengan **2,527 foto** untuk mengecek kepintarannya
- ⏱️ Waktu pelatihan: ~1 jam (menggunakan komputer super kencang di cloud)

**Hasilnya?**
- Akurasi: 89.91% ✅
- Presisi: 90% (jarang salah tebak)
- Recall: 90% (jarang kelewatan)

---

## 🛠️ Teknologi yang Dipakai

Untuk yang penasaran teknisnya:

| Komponen | Teknologi |
|----------|-----------|
| Otak AI | CNN (Convolutional Neural Network) |
| Framework | TensorFlow |
| Jumlah Parameter | 9.8 juta (seberapa "pintar" AI-nya) |
| Antarmuka Web | Streamlit |
| Bahasa Pemrograman | Python |
| Pelatihan | Google Colab (GPU gratis) |

---

## 💻 Cara Pakai (Online)

**Paling Mudah:** Langsung buka link di atas!

Atau kalau mau jalankan di komputer sendiri:

### 1. Install Python & Library
```bash
pip install tensorflow streamlit pillow opencv-python gdown
```

### 2. Download Code
```bash
git clone https://github.com/farellthealchemist/waste-classification-project.git
cd waste-classification-project
```

### 3. Jalankan Aplikasi
```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

## 📁 Isi Project
```
📦 waste-classification-project/
├── 📄 app.py                    # Kode aplikasi web
├── 📄 README.md                 # File yang sedang kamu baca ini
├── 📄 requirements.txt          # Daftar library yang dibutuhkan
├── 📄 screenshot.png            # Screenshot aplikasi
├── 📂 models/                   # Folder untuk "otak" AI
│   └── best_model.keras         # File AI yang sudah dilatih (37 MB)
├── 📂 notebooks/                # Script untuk persiapan data
│   ├── explore_data.py
│   ├── data_generator.py
│   ├── visualize_data.py
│   └── ... (file lainnya)
└── 📂 dataset/                  # Foto-foto untuk latihan (tidak di-upload)
    ├── TRAIN/                   # 22,564 foto untuk latihan
    └── TEST/                    # 2,527 foto untuk ujian
```

---

## 🎓 Proses Pembuatan

Project ini dikerjakan dalam **3 tahap besar:**

### **Tahap 1: Persiapan (2 minggu)**
- Download 25,000+ foto sampah
- Pisahkan jadi data latihan & ujian
- Ubah ukuran foto jadi seragam (224x224 pixel)
- Tambah variasi foto (putar, zoom, flip)

### **Tahap 2: Melatih AI (1 minggu)**
- Ajarin AI mengenali sampah (pakai Google Colab)
- AI "belajar" dari 22,564 foto
- Tes AI dengan 2,527 foto
- Dapat akurasi 89.91%! 🎉

### **Tahap 3: Bikin Website (1 minggu)**
- Buat tampilan web yang gampang dipakai
- Upload foto → AI prediksi → tampil hasil
- Deploy ke internet (gratis di Streamlit Cloud)

**Total waktu:** ~1 bulan

---

## 📸 Tampilan Aplikasi

### Homepage
![Screenshot Aplikasi](screenshot.png)

**Fitur-fitur:**
- ✅ Upload foto dengan drag & drop
- ✅ Hasil prediksi real-time (cepat!)
- ✅ Progress bar keyakinan AI (99%+)
- ✅ Info cara buang sampah yang benar
- ✅ Download hasil klasifikasi
- ✅ Tampilan modern & user-friendly

---

## 🎯 Manfaat Aplikasi

**Untuk Masyarakat:**
- Bantu pilah sampah dengan benar
- Edukasi cara buang sampah yang tepat
- Gratis & mudah diakses

**Untuk Lingkungan:**
- Kurangi sampah yang salah tempat
- Tingkatkan daur ulang
- Bantu jaga bumi kita 🌍

**Untuk Pelajar/Mahasiswa:**
- Contoh project AI yang aplikatif
- Belajar Computer Vision dengan mudah
- Portfolio untuk CV/interview

---

## ❓ FAQ (Pertanyaan yang Sering Ditanya)

**Q: Apakah gratis?**
A: Ya! 100% gratis untuk dipakai.

**Q: Apakah data foto saya disimpan?**
A: Tidak. Foto hanya diproses sementara dan langsung dihapus.

**Q: Bisa offline?**
A: Bisa! Download code-nya dan jalankan di laptop sendiri.

**Q: Akurat untuk semua jenis sampah?**
A: Aplikasi ini dilatih khusus untuk 2 kategori (Organic & Recyclable). Akurasi 89.91% untuk kategori ini.

**Q: Bisa tambah kategori lain?**
A: Bisa! Tapi perlu latih ulang AI dengan dataset baru.

---

## 🚀 Pengembangan Selanjutnya

**Rencana update di masa depan:**
- [ ] Tambah kategori: Elektronik, B3, dll
- [ ] Prediksi multiple foto sekaligus
- [ ] Aplikasi mobile (Android/iOS)
- [ ] Integrasi dengan tempat sampah pintar
- [ ] Multi-bahasa (English, Indonesia, dll)

---

## 👨‍💻 Pembuat

**Farell Adrian**
- Project: Computer Vision untuk Pemula
- Kontak: [GitHub](https://github.com/farellthealchemist)

---

## 📄 Lisensi

Project ini dibuat untuk **tujuan edukasi**.
Bebas dipakai dan dikembangkan untuk belajar! 📚

---

## 🙏 Terima Kasih

Terima kasih sudah melihat project ini!

Kalau ada pertanyaan atau saran, jangan ragu untuk:
- Buka issue di GitHub
- Atau kontak langsung

**Mari sama-sama jaga lingkungan! 🌱**

---

<div align="center">
  <p>Made with ❤️ and ☕ by Farell</p>
  <p>Powered by TensorFlow & Streamlit</p>
</div>
