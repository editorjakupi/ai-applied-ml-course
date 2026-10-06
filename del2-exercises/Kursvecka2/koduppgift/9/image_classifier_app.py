
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import pandas as pd
import matplotlib.pyplot as plt

# Konfigurera Streamlit
st.set_page_config(
    page_title="Avancerad Bildklassificering",
    page_icon="",
    layout="wide"
)

@st.cache_resource
def load_model():
    return ResNet50(weights='imagenet')

def preprocess_image(uploaded_file):
    img_pil = Image.open(uploaded_file)
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    img_resized = img_pil.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    return img_preprocessed, img_pil

def predict_image(model, img_preprocessed, top_n=5):
    predictions = model.predict(img_preprocessed, verbose=0)
    decoded = decode_predictions(predictions, top=top_n)[0]
    return decoded

def main():
    st.title(" Avancerad Bildklassificering")
    st.markdown("**Klassificera bilder med ResNet50 - Förtränad CNN-modell**")

    # Sidebar med inställningar
    with st.sidebar:
        st.header(" Inställningar")
        top_n = st.slider("Antal topprediktioner att visa", 3, 10, 5)
        show_details = st.checkbox("Visa detaljerad information", value=True)

        st.markdown("---")
        st.header("ℹ Om applikationen")
        st.info("""
        Denna applikation använder ResNet50, en förtränad CNN-modell.
        Modellen kan klassificera 1000 olika klasser från ImageNet.
        """)

    # Ladda modell
    with st.spinner('Laddar modell...'):
        model = load_model()
    st.success(' Modell laddad!')

    st.markdown("---")

    # Huvudfunktionalitet
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header(" Ladda upp bild")
        uploaded_file = st.file_uploader(
            "Välj en bildfil",
            type=['png', 'jpg', 'jpeg'],
            help="Ladda upp en bild för klassificering"
        )

    if uploaded_file is not None:
        # Visa bild
        img_pil = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(" Originalbild")
            st.image(img_pil, caption="Uppladdad bild", use_container_width=True)
            st.caption(f"Storlek: {img_pil.size[0]}x{img_pil.size[1]} pixlar")

        # Prediktera
        with st.spinner('Analyserar bild...'):
            img_preprocessed, _ = preprocess_image(uploaded_file)
            predictions = predict_image(model, img_preprocessed, top_n=top_n)

        with col2:
            st.subheader(" Prediktionsresultat")

            # Visa topprediktioner med progress bars
            for i, (_, label, score) in enumerate(predictions, 1):
                st.write(f"**{i}. {label}**")
                st.progress(float(score))
                st.caption(f"{score:.4f} ({score*100:.2f}%)")

        # Detaljerad information
        if show_details:
            st.markdown("---")

            # Tabell
            st.subheader(" Detaljerad tabell")
            results_df = pd.DataFrame([
                {'Rank': i+1, 'Klass': label, 'Sannolikhet': f"{score:.4f}", 'Procent': f"{score*100:.2f}%"}
                for i, (_, label, score) in enumerate(predictions)
            ])
            st.dataframe(results_df, use_container_width=True)

            # Visualisering
            st.subheader(" Sannolikhetsfördelning")
            fig, ax = plt.subplots(figsize=(10, 6))
            labels = [pred[1] for pred in predictions]
            scores = [pred[2] for pred in predictions]
            ax.barh(labels, scores)
            ax.set_xlabel('Sannolikhet')
            ax.set_title('Topp prediktioner')
            plt.tight_layout()
            st.pyplot(fig)

        # Bästa prediktion
        best_pred = predictions[0]
        st.success(f" **Bästa prediktion:** {best_pred[1]} ({best_pred[2]*100:.2f}%)")

if __name__ == "__main__":
    main()
