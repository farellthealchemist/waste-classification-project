import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Waste Classification",
    page_icon="🗑️",
    layout="centered"
)

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    model_path = 'models/best_model.keras'
    model = tf.keras.models.load_model(model_path)
    return model

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