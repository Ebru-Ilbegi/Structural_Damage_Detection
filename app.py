import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Sayfa Ayarlari / Page Configuration
st.set_page_config(page_title="Yapisal Hasar Analizi / Structural Damage Analysis", layout="wide")

# Modelleri Onbellege Alma (Hiz icin kritik) / Caching Models (Critical for speed)
@st.cache_resource
def load_models():
    model_crack = tf.keras.models.load_model('crack_model.keras')
    model_multi = tf.keras.models.load_model('multi_task_model.keras')
    model_sdnet = tf.keras.models.load_model('sdnet_component_model.keras')
    return model_crack, model_multi, model_sdnet

model_crack, model_multi, model_sdnet = load_models()

# Ana Baslik ve Aciklama / Main Title and Description
st.title("Yapisal Catlak ve Hasar Analiz Sistemi / Structural Crack and Damage Analysis System")
st.write("Lutfen analiz etmek istediginiz yapi gorselini yukleyin. / Please upload the structural image you want to analyze.")

# Dosya Yukleme Alani / File Upload Area
uploaded_file = st.file_uploader("Bir fotograf secin / Select an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Resmi Acma ve Ekranda Gosterme / Opening and Displaying the Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Yuklenen Gorsel / Uploaded Image", width=400)
    
    # Analizi Baslatma Butonu / Start Analysis Button
    if st.button("Analizi Baslat / Start Analysis"):
        with st.spinner('Modeller gorseli inceliyor... / Models are analyzing the image...'):
            
            # 1. Resmi Modele Hazirlama 
            img_resized = image.resize((224, 224))
            
            # DİKKAT: Crack modelinin doğru çalışması için / 255.0'i kaldırdık
            img_array = tf.keras.utils.img_to_array(img_resized) 
            img_array_batch = np.expand_dims(img_array, axis=0)

            # EĞER diğer 2 model (SDNET/Multi) /255.0 ile eğitildiyse diye yedek:
            img_array_normalized = img_array_batch / 255.0

            st.subheader("Analiz Sonuclari / Analysis Results")
            
            # --- 1. GENEL ÇATLAK MODELİ (Düzeltildi) ---
            crack_pred = model_crack.predict(img_array_batch)
            ham_skor = crack_pred[0][0] 
            
            # Test kodundaki Sigmoid dönüşümü eklendi
            crack_probability = tf.math.sigmoid(ham_skor).numpy() 
            
            catlak_var = crack_probability > 0.5
            
            if catlak_var:
                catlak_durumu = "Catlak Tespit Edildi / Crack Detected"
                eminlik = crack_probability * 100
                st.error(f"⚠️ Temel Durum / Basic Status: {catlak_durumu} (Eminlik: %{eminlik:.2f})")
            else:
                catlak_durumu = "Temiz / Clean"
                eminlik = (1 - crack_probability) * 100
                st.success(f"✅ Temel Durum / Basic Status: {catlak_durumu} (Eminlik: %{eminlik:.2f})")

            # --- 2. SDNET MODELİ ---
            # Not: Eğer SDNET saçmalarsa 'img_array_batch' yerine 'img_array_normalized' yazın
            sdnet_pred = model_sdnet.predict(img_array_batch)
            comp_labels = ['Decks (Doseme/Deck)', 'Pavements (Kaldirim/Pavement)', 'Walls (Duvar/Wall)']
            predicted_index = np.argmax(sdnet_pred[0]) 
            secilen_yapi = comp_labels[predicted_index] 
            st.info(f"Yapi Elemani / Structural Component: {secilen_yapi}")

            # --- 3. MULTI-TASK MODELİ ---
            if catlak_var:
                # Not: Eğer Genişlik Modeli saçmalarsa 'img_array_batch' yerine 'img_array_normalized' yazın
                multi_pred = model_multi.predict(img_array_batch)
                tahmini_genislik = multi_pred[2][0][0] 
                tahmini_genislik = max(0.0, tahmini_genislik) 
                st.warning(f"Tahmini Catlak Genisligi / Estimated Crack Width: {tahmini_genislik:.2f} mm")
            else:
                st.write("Catlak tespit edilmedi, genislik hesaplanmadi.")