import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Sidkonfiguration
st.set_page_config(page_title="Bilprisprediktor", page_icon="", layout="wide")

# Ladda modell
@st.cache_resource
def load_model():
    try:
        return joblib.load('car_pipeline.pkl')
    except:
        st.error("Kunde inte ladda modellen. Kör notebooken först!")
        return None

# Pipeline-klass för appen
class CarPipeline:
    def __init__(self):
        self.model = None
        self.brand_encoder = OneHotEncoder(drop='first', sparse_output=False)
        self.model_encoder = OneHotEncoder(drop='first', sparse_output=False)
        self.fuel_encoder = OneHotEncoder(drop='first', sparse_output=False)
        self.transmission_encoder = OneHotEncoder(drop='first', sparse_output=False)
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        X_processed = X.copy()
        X_processed['Age'] = 2024 - X_processed['Year']
        X_processed['Miles_Per_Year'] = X_processed['Mileage'] / (X_processed['Age'] + 1)
        
        self.brand_encoder.fit(X_processed[['Brand']])
        self.model_encoder.fit(X_processed[['Model']])
        self.fuel_encoder.fit(X_processed[['Fuel_Type']])
        self.transmission_encoder.fit(X_processed[['Transmission']])
        
        brand_encoded = self.brand_encoder.transform(X_processed[['Brand']])
        model_encoded = self.model_encoder.transform(X_processed[['Model']])
        fuel_encoded = self.fuel_encoder.transform(X_processed[['Fuel_Type']])
        transmission_encoded = self.transmission_encoder.transform(X_processed[['Transmission']])
        
        numeric_features = ['Year', 'Mileage', 'Engine_Size', 'Doors', 'Owner_Count', 'Age', 'Miles_Per_Year']
        X_numeric = X_processed[numeric_features]
        X_scaled = self.scaler.fit_transform(X_numeric)
        
        X_combined = np.hstack([X_scaled, brand_encoded, model_encoded, fuel_encoded, transmission_encoded])
        self.model.fit(X_combined, y)
        return self
    
    def predict(self, X):
        X_processed = X.copy()
        X_processed['Age'] = 2024 - X_processed['Year']
        X_processed['Miles_Per_Year'] = X_processed['Mileage'] / (X_processed['Age'] + 1)
        
        brand_encoded = self.brand_encoder.transform(X_processed[['Brand']])
        model_encoded = self.model_encoder.transform(X_processed[['Model']])
        fuel_encoded = self.fuel_encoder.transform(X_processed[['Fuel_Type']])
        transmission_encoded = self.transmission_encoder.transform(X_processed[['Transmission']])
        
        numeric_features = ['Year', 'Mileage', 'Engine_Size', 'Doors', 'Owner_Count', 'Age', 'Miles_Per_Year']
        X_numeric = X_processed[numeric_features]
        X_scaled = self.scaler.transform(X_numeric)
        
        X_combined = np.hstack([X_scaled, brand_encoded, model_encoded, fuel_encoded, transmission_encoded])
        return self.model.predict(X_combined)

# Huvudapp
def main():
    st.title(" Bilprisprediktor")
    st.write("Prediktera bilpriser baserat på bilens egenskaper")
    
    # Ladda modell
    pipeline = load_model()
    if pipeline is None:
        return
    
    # Ladda data för dropdown-värden
    try:
        df = pd.read_csv('car_price_dataset.csv', sep=';')
        brands = sorted(df['Brand'].unique())
        models = sorted(df['Model'].unique())
        fuel_types = sorted(df['Fuel_Type'].unique())
        transmissions = sorted(df['Transmission'].unique())
    except:
        st.error("Kunde inte ladda data. Kontrollera att car_price_dataset.csv finns!")
        return
    
    # Input-sektion
    st.header(" Bilinformation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox("Märke", brands, help="Välj bilmärke")
        model = st.selectbox("Modell", models, help="Välj modell")
        year = st.slider("År", 1990, 2024, 2020, help="Bilens tillverkningsår")
        mileage = st.number_input("Miltal", min_value=0, max_value=500000, value=50000, help="Antal körda mil")
    
    with col2:
        engine_size = st.slider("Motorstorlek", 0.5, 8.0, 2.0, step=0.1, help="Motorstorlek i liter")
        doors = st.selectbox("Antal dörrar", [2, 3, 4, 5], help="Antal dörrar")
        fuel_type = st.selectbox("Bränsletyp", fuel_types, help="Typ av bränsle")
        transmission = st.selectbox("Växellåda", transmissions, help="Typ av växellåda")
        owner_count = st.number_input("Antal ägare", min_value=1, max_value=10, value=1, help="Antal tidigare ägare")
    
    # Prediktion
    if st.button(" Prediktera pris", type="primary"):
        # Skapa input data
        input_data = pd.DataFrame({
            'Brand': [brand],
            'Model': [model],
            'Year': [year],
            'Mileage': [mileage],
            'Engine_Size': [engine_size],
            'Doors': [doors],
            'Fuel_Type': [fuel_type],
            'Transmission': [transmission],
            'Owner_Count': [owner_count]
        })
        
        # Prediktera
        prediction = pipeline.predict(input_data)[0]
        
        # Visa resultat
        st.header(" Predikterat pris")
        st.metric("Pris", f"${prediction:,.0f}")
        
        # Visa använd input
        st.subheader(" Använd information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Märke:** {brand}")
            st.write(f"**Modell:** {model}")
            st.write(f"**År:** {year}")
            st.write(f"**Miltal:** {mileage:,}")
        with col2:
            st.write(f"**Motor:** {engine_size}L")
            st.write(f"**Dörrar:** {doors}")
            st.write(f"**Bränsle:** {fuel_type}")
            st.write(f"**Växellåda:** {transmission}")
    
    # Om appen
    with st.expander("ℹ Om denna app"):
        st.write("""
        **Bilprisprediktor** använder Machine Learning för att prediktera bilpriser.
        
        **Modell:** Random Forest Regressor
        **Prestanda:** RMSE ~$481 (genomsnittligt fel)
        **Features:** År, miltal, motorstorlek, märke, modell, bränsletyp, växellåda, antal dörrar, antal ägare
        
        **Hur det fungerar:**
        1. Ange bilens egenskaper
        2. Klicka på "Prediktera pris"
        3. Få en uppskattning av bilens värde
        """)

if __name__ == "__main__":
    main()

