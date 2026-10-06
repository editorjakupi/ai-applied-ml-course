import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import joblib
import os

# Scikit-learn komponenter
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

# Konfigurera Streamlit
st.set_page_config(
    page_title="MNIST Siffra Klassificering",
    page_icon="",
    layout="wide"
)

def preprocess_image(image, scaler):
    """
    Preprocessing som matchar notebook-exakt
    Använder samma StandardScaler som modellen tränades på
    
    Args:
        image: PIL Image eller numpy array
        scaler: StandardScaler som användes vid träning
    
    Returns:
        numpy array: Förbearbetad bild som 1D array (784 pixlar) - SKALAD MED STANDARDSCALER
    """
    try:
        # Konvertera till numpy
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        # Konvertera till gråskala
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        original_img = img.copy()
        
        # Resize till 28x28 (samma som MNIST)
        if img.shape != (28, 28):
            img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img
        
        # Konvertera till float (0-255 range, INTE 0-1!)
        img_normalized = img_resized.astype(float)
        
        # Konvertera till 1D array (samma format som MNIST)
        img_vector = img_normalized.reshape(1, -1)
        
        # KRITISKT: Använd samma StandardScaler som modellen tränades på!
        img_vector_scaled = scaler.transform(img_vector)
        
        return img_vector_scaled, original_img, img_normalized.reshape(28, 28)
        
    except Exception as e:
        st.error(f" Fel vid preprocessing: {e}")
        return None, None, None

def load_model():
    """Ladda tränad modell"""
    model_file = 'mnist_model.pkl'
    if os.path.exists(model_file):
        return joblib.load(model_file)
    return None

def main():
    st.title(" MNIST Siffra Klassificering")
    st.markdown("**Klassificera handskrivna siffror med Extra Trees modell**")
    
    # Kontrollera om modell finns
    model_data = load_model()
    
    if model_data is None:
        st.warning(" Ingen tränad modell hittades. Kör Jupyter notebook först!")
        st.info(" Kör alla celler i `Kodexempel_1_MNIST_Klassificering.ipynb` för att träna modellen.")
        return
    
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Visa modellinformation
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test Accuracy", f"{model_data['test_accuracy']:.4f}")
    with col2:
        st.metric("Val Accuracy", f"{model_data.get('val_accuracy', model_data['test_accuracy']):.4f}")
    with col3:
        st.metric("Dataset storlek", f"{model_data['dataset_size']:,}")
    with col4:
        st.metric("Modelltyp", "Extra Trees")
    
    st.markdown("---")
    
    # Huvudfunktionalitet
    st.header(" Ladda upp din siffra")
    
    # Alternativ för input
    input_method = st.radio(
        "Välj input-metod:",
        [" Ladda upp bild", " Testa med MNIST-bild"]
    )
    
    if input_method == " Ladda upp bild":
        uploaded_file = st.file_uploader(
            "Välj en bildfil (PNG, JPG, JPEG)",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            # Ladda bild
            image = Image.open(uploaded_file)
            
            # Visa originalbild
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(" Originalbild")
                st.image(image, caption="Uppladdad bild", use_container_width=True)
                st.info(f" Originalbild: {image.size[0]}x{image.size[1]} pixlar")
            
            # Preprocessa och prediktera
            img_vector, original_img, processed_img = preprocess_image(image, scaler)
            
            if img_vector is not None:
                with col2:
                    st.subheader(" Förbearbetad bild")
                    # Normalisera bilden för visning (0-1 range)
                    processed_img_display = processed_img / 255.0
                    st.image(processed_img_display, caption="28x28 gråskala", use_container_width=True)
                
                # Prediktera
                prediction = model.predict(img_vector)[0]
                probability = model.predict_proba(img_vector)[0]
                max_prob = np.max(probability)
                
                # Visa resultat
                st.markdown("---")
                st.header(" Prediktionsresultat")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predikterad siffra", f"**{prediction}**")
                with col2:
                    st.metric("Sannolikhet", f"{max_prob:.3f}")
                with col3:
                    confidence = "Hög" if max_prob > 0.8 else "Medium" if max_prob > 0.5 else "Låg"
                    st.metric("Tillförlitlighet", confidence)
                
                # Visa sannolikheter för alla siffror
                st.subheader(" Sannolikheter för alla siffror")
                prob_df = pd.DataFrame({
                    'Siffra': range(10),
                    'Sannolikhet': probability
                })
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(prob_df['Siffra'], prob_df['Sannolikhet'])
                ax.set_xlabel('Siffra')
                ax.set_ylabel('Sannolikhet')
                ax.set_title('Sannolikheter för alla siffror')
                ax.set_ylim(0, 1)
                
                # Färgkoda den predikterade siffran
                bars[prediction].set_color('red')
                
                st.pyplot(fig)
    
    elif input_method == " Testa med MNIST-bild":
        if st.button(" Välj slumpmässig MNIST-bild"):
            # Ladda MNIST för test
            mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
            X_test = mnist["data"][20000:21000]  # Ta testdata
            y_test = mnist["target"][20000:21000].astype(np.uint8)
            
            # Välj slumpmässig bild
            idx = np.random.randint(0, len(X_test))
            test_image = X_test[idx].reshape(28, 28)
            true_label = y_test[idx]
            
            # Visa bilder
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(" Slumpmässig MNIST-bild")
                st.image(test_image, caption=f"Sann etikett: {true_label}", use_container_width=True)
            
            # Preprocessa och prediktera
            img_vector, _, processed_img = preprocess_image(test_image, scaler)
            
            if img_vector is not None:
                with col2:
                    st.subheader(" Förbearbetad bild")
                    # Normalisera bilden för visning (0-1 range)
                    processed_img_display = processed_img / 255.0
                    st.image(processed_img_display, caption="Preprocessad", use_container_width=True)
                
                # Prediktera
                prediction = model.predict(img_vector)[0]
                probability = model.predict_proba(img_vector)[0]
                max_prob = np.max(probability)
                
                # Visa resultat
                st.markdown("---")
                st.header(" Prediktionsresultat")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sann etikett", f"**{true_label}**")
                with col2:
                    st.metric("Predikterad", f"**{prediction}**")
                with col3:
                    st.metric("Sannolikhet", f"{max_prob:.3f}")
                with col4:
                    correct = " KORREKT" if prediction == true_label else " FEL"
                    st.metric("Resultat", correct)
    
    # Tips
    st.markdown("---")
    st.header(" Tips för bästa resultat")
    st.markdown("""
    **För egna bilder:**
    -  Använd svart text på vit bakgrund (VIKTIGT!)
    -  Rita siffran i mitten av bilden
    -  Gör siffran tydlig och stor (minst 50x50 pixlar)
    -  Undvik dekorationer och extra linjer
    -  En siffra per bild
    -  Använd tydlig, enkel stil (som MNIST-siffror)
    -  INTE vit text på svart bakgrund (det fungerar inte!)
    
    **Bildstorlek:**
    - Appen hanterar alla bildstorlekar automatiskt
    - Bilder resizas till 28x28 pixlar för klassificering
    - Större bilder ger oftast bättre resultat
    """)

if __name__ == "__main__":
    main()