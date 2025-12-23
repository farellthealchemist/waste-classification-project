import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from datetime import datetime

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Waste Classification",
    page_icon="🗑️",
    layout="centered"
)

# ============================================
# LOAD MODEL - WITH ERROR HANDLING
# ============================================
import os
import gdown

@st.cache_resource
def load_model():
    try:
        model_path = 'models/best_model.keras'
        
        # Check if model exists locally
        if not os.path.exists(model_path):
            st.info("📥 Downloading model from Google Drive...")
            os.makedirs('models', exist_ok=True)
            
            # Google Drive file ID
            file_id = "1vI88ct35fkZcE86UCWVIpSxEkpAvozjS"
            url = f"https://drive.google.com/uc?id={file_id}"
            
            # Download model
            gdown.download(url, model_path, quiet=False)
            st.success("✅ Model downloaded successfully!")
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("💡 Pastikan model sudah diupload ke Google Drive dan link-nya public")
        st.stop()

model = load_model()

# ============================================
# FUNCTIONS
# ============================================
def preprocess_image(image):
    """Preprocess image untuk prediksi"""
    # Resize ke 224x224
    img = cv2.resize(image, (224, 224))
    # Convert BGR ke RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalisasi
    img = img / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def predict(image):
    """Prediksi jenis sampah"""
    # Preprocess
    processed_img = preprocess_image(image)
    
    # Prediksi
    prediction = model.predict(processed_img, verbose=0)
    confidence = float(prediction[0][0])
    
    # 0 = Organic, 1 = Recyclable
    if confidence > 0.5:
        label = "Recyclable (Daur Ulang)"
        emoji = "♻️"
        color = "blue"
        info = """
        **Sampah Recyclable** dapat didaur ulang!
        
        Contoh:
        - Plastik (botol, kemasan)
        - Kertas & karton
        - Kaleng & logam
        - Kaca
        
        💡 **Cara Buang:**
        - Bersihkan sebelum dibuang
        - Buang ke tempat sampah KUNING/BIRU
        - Atau serahkan ke bank sampah
        """
    else:
        label = "Organic (Organik)"
        emoji = "🌿"
        color = "green"
        info = """
        **Sampah Organik** berasal dari makhluk hidup.
        
        Contoh:
        - Sisa makanan
        - Kulit buah & sayur
        - Daun & ranting
        - Tulang
        
        💡 **Cara Buang:**
        - Buang ke tempat sampah HIJAU
        - Bisa dijadikan kompos
        - Bisa untuk pakan ternak
        """
    
    return label, confidence, emoji, color, info

# ============================================
# HEADER
# ============================================
st.title("🗑️ Sistem Klasifikasi Sampah")
st.markdown("### Berbasis AI - Computer Vision")
st.markdown("---")

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini menggunakan **Deep Learning (CNN)** 
    untuk mengklasifikasikan sampah menjadi:
    
    - 🌿 **Organic** (Organik)
    - ♻️ **Recyclable** (Daur Ulang)
    
    **Akurasi Model:** 89.91%
    
    **Total Training Data:** 22,564 gambar
    """)
    
    st.markdown("---")
    st.markdown("**📊 Model Performance:**")
    st.markdown("- Precision: 90%")
    st.markdown("- Recall: 90%")
    st.markdown("- F1-Score: 90%")
    
    st.markdown("---")
    st.caption("Dibuat oleh: Farell Adrian")
    st.caption("Project: Computer Vision untuk Pemula")

# ============================================
# MAIN CONTENT
# ============================================

# Upload Image
uploaded_file = st.file_uploader(
    "📤 Upload Foto Sampah",
    type=['jpg', 'jpeg', 'png'],
    help="Upload foto sampah yang ingin diklasifikasikan"
)

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Foto yang Diupload")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Hasil Prediksi")
        
        # Convert PIL to OpenCV
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Predict
        with st.spinner('🤔 Menganalisis gambar...'):
            label, confidence, emoji, color, info = predict(img_bgr)
        
        # Display result
        st.markdown(f"## {emoji} **{label}**")
        
        # Confidence bar
        if label.startswith("Recyclable"):
            conf_display = confidence * 100
        else:
            conf_display = (1 - confidence) * 100
            
        st.progress(conf_display / 100)
        st.markdown(f"**Tingkat Keyakinan:** {conf_display:.2f}%")
    
    # Info section
    st.markdown("---")
    st.markdown(info)
    
    # Download Button untuk Save Result
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Prepare result text
        result_text = f"""
HASIL KLASIFIKASI SAMPAH
========================

Kategori      : {label}
Confidence    : {conf_display:.2f}%
Tanggal       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Nama File     : {uploaded_file.name}

Informasi:
{info.strip()}

---
Waste Classification System
Powered by TensorFlow & Streamlit
        """
        
        st.download_button(
            label="📥 Download Hasil Klasifikasi",
            data=result_text,
            file_name=f"hasil_klasifikasi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # Helpful message
    st.markdown("---")
    st.info("💡 **Tip:** Upload gambar lain dengan klik 'Browse files' atau drag & drop untuk klasifikasi baru")

else:
    # Instructions
    st.info("👆 Upload foto sampah untuk memulai klasifikasi")
    
    # Example images section
    st.markdown("---")
    st.subheader("💡 Contoh Gambar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌿 Organic")
        st.markdown("""
        - Sisa makanan
        - Kulit buah
        - Daun & ranting
        - Sayuran busuk
        """)
    
    with col2:
        st.markdown("### ♻️ Recyclable")
        st.markdown("""
        - Botol plastik
        - Kertas & kardus
        - Kaleng minuman
        - Kemasan makanan
        """)
    
    # Contoh Hasil Klasifikasi (FEATURE #2)
    st.markdown("---")
    st.subheader("📸 Contoh Hasil Klasifikasi")
    st.markdown("Berikut adalah contoh hasil prediksi model:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("**🌿 Organic**")
        st.markdown("**Confidence:** 99.77%")
        st.caption("✅ Sayuran, buah-buahan, sisa makanan")
        st.caption("Model dapat mengidentifikasi sampah organik dengan tingkat akurasi sangat tinggi")
    
    with col2:
        st.info("**♻️ Recyclable**")
        st.markdown("**Confidence:** 99.33%")
        st.caption("✅ Plastik, kertas, kaleng, botol")
        st.caption("Model dapat membedakan material yang dapat didaur ulang")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Waste Classification System | Powered by TensorFlow & Streamlit</p>
    <p>Membantu memilah sampah dengan lebih mudah dan cepat 🌍</p>
</div>
""", unsafe_allow_html=True)