import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Konfigurera sidan
st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="",
    layout="wide"
)

# Definiera pipeline-klass (samma som i notebooken)
class DiamondPipeline:
    def __init__(self, model=None):
        self.model = model  # Ingen default modell
        self.encoder = OneHotEncoder(drop='first', sparse_output=False)
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        # Separera numeriska och kategoriska features
        numeric_features = ['carat', 'depth', 'table', 'x', 'y', 'z', 'volume']
        categorical_features = ['cut', 'color', 'clarity']
        
        X_num = X[numeric_features]
        X_cat = X[categorical_features]
        
        # Skala numeriska features
        X_num_scaled = self.scaler.fit_transform(X_num)
        
        # Encode kategoriska features till dummy variables
        X_cat_encoded = self.encoder.fit_transform(X_cat)
        
        # Kombinera features
        X_processed = np.hstack([X_num_scaled, X_cat_encoded])
        
        # Träna modellen
        self.model.fit(X_processed, y)
        return self
    
    def predict(self, X):
        # Samma preprocessing som vid träning
        numeric_features = ['carat', 'depth', 'table', 'x', 'y', 'z', 'volume']
        categorical_features = ['cut', 'color', 'clarity']
        
        X_num = X[numeric_features]
        X_cat = X[categorical_features]
        
        X_num_scaled = self.scaler.transform(X_num)
        X_cat_encoded = self.encoder.transform(X_cat)
        
        # Kombinera och prediktera
        X_processed = np.hstack([X_num_scaled, X_cat_encoded])
        return self.model.predict(X_processed)

# Header
st.title("Diamond Price Predictor")
st.markdown("Prediktera diamantpriser baserat på diamantens egenskaper")

# Ladda pipeline
@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load('diamond_pipeline.pkl')
        return pipeline
    except FileNotFoundError:
        st.error("Pipeline-filen hittades inte! Kör först notebooken för att skapa modellen.")
        return None
    except Exception as e:
        st.error(f"Fel vid laddning av pipeline: {str(e)}")
        return None

# Ladda modell
pipeline = load_pipeline()

if pipeline is None:
    st.stop()

# Skapa tre kolumner
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("### Ange diamantens egenskaper")
    
    # Skapa två rader för input
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        carat = st.number_input("Carat (vikt)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        cut = st.selectbox("Cut (slipning)", ["Ideal", "Premium", "Very Good", "Good", "Fair"])
        color = st.selectbox("Color (färg)", ["D", "E", "F", "G", "H", "I", "J"])
        clarity = st.selectbox("Clarity (klarhet)", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])
    
    with row1_col2:
        depth = st.number_input("Depth (%)", min_value=40.0, max_value=80.0, value=61.5, step=0.1)
        table = st.number_input("Table (%)", min_value=40.0, max_value=100.0, value=55.0, step=0.1)
        x = st.number_input("Length (x)", min_value=0.0, max_value=20.0, value=5.7, step=0.1)
        y = st.number_input("Width (y)", min_value=0.0, max_value=20.0, value=5.7, step=0.1)
        z = st.number_input("Depth (z)", min_value=0.0, max_value=20.0, value=3.5, step=0.1)

with col2:
    st.markdown("### Beräknade värden")
    
    # Beräkna volym
    volume = x * y * z
    
    st.metric("Volym", f"{volume:.2f} mm³")
    st.metric("Dimensioner", f"{x}×{y}×{z} mm")

with col3:
    st.markdown("### Information")
    
    with st.expander("Om diamantkvalitet", expanded=False):
        st.markdown("""
        **Cut (Slipning):**
        - Ideal: Perfekt proportioner
        - Premium: Mycket bra proportioner
        - Very Good: Bra proportioner
        - Good: Acceptabla proportioner
        - Fair: Mindre bra proportioner
        
        **Color (Färg):**
        - D: Färglös (bästa)
        - E: Nästan färglös
        - F: Färglös
        - G: Nästan färglös
        - H: Lätt färgad
        - I: Lätt färgad
        - J: Ljulgul
        
        **Clarity (Klarhet):**
        - IF: Internally Flawless
        - VVS1/VVS2: Very Very Slightly Included
        - VS1/VS2: Very Slightly Included
        - SI1/SI2: Slightly Included
        - I1: Included
        """)

# Prediktion
st.markdown("---")
if st.button("Prediktera pris", type="primary", use_container_width=True):
    try:
        # Skapa input data
        input_data = pd.DataFrame({
            'carat': [carat], 'cut': [cut], 'color': [color], 'clarity': [clarity],
            'depth': [depth], 'table': [table], 'x': [x], 'y': [y], 'z': [z],
            'volume': [volume]
        })
        
        # Prediktera
        predicted_price = pipeline.predict(input_data)[0]
        
        # Visa resultat
        st.markdown(f"## Predikterat pris: **${predicted_price:,.0f}**")
        
        # Skapa två kolumner för detaljerad information
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown("### Grundegenskaper")
            st.metric("Carat", f"{carat}")
            st.metric("Cut", cut)
            st.metric("Color", color)
            st.metric("Clarity", clarity)
        
        with info_col2:
            st.markdown("### Mått")
            st.metric("Depth", f"{depth}%")
            st.metric("Table", f"{table}%")
            st.metric("Volume", f"{volume:.2f} mm³")
            st.metric("Predikterat pris", f"${predicted_price:,.0f}")
        
    except Exception as e:
        st.error(f"Fel vid prediktion: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Diamond Price Predictor | ML-modell baserad på diamantens egenskaper*")
