#  Bilprisprediktion - Fråga 14

En enkel och effektiv ML-lösning för bilprisprediktion med Streamlit-app.

##  Filer

- **`Fråga_14_Förbättrad.ipynb`** - Jupyter notebook med ML-modell
- **`car_price_app.py`** - Streamlit-app för prediktion
- **`car_pipeline.pkl`** - Sparad modell (genereras av notebooken)
- **`car_price_dataset.csv`** - Dataset med bilinformation

##  Användning

### 1. Träna modellen

```bash
# Kör notebooken för att träna och spara modellen
jupyter notebook Fråga_14_Förbättrad.ipynb
```

### 2. Starta Streamlit-appen

```bash
streamlit run car_price_app.py
```

##  Förbättringar

### Notebook (`Fråga_14_Förbättrad.ipynb`)

-  **Enkel och koncis** - Bara 6 celler
-  **OneHotEncoder** - Korrekt hantering av kategoriska variabler
-  **Endast RMSE** - Fokuserad på felmått
-  **Inga onödiga visualiseringar** - Bara nödvändig EDA
-  **En pipeline-klass** - Allt i en fil

### Streamlit-app (`car_price_app.py`)

-  **Mycket enkel** - Bara en sida
-  **Användarvänlig** - Tydlig input/output
-  **Inga onödiga grafer** - Fokuserad på prediktion
-  **Korrekt encoding** - OneHotEncoder
-  **Snabb laddning** - Minimal kod

##  Teknisk implementation

### ML-modell

- **Algoritm:** Random Forest Regressor
- **Encoding:** OneHotEncoder (dummy variables)
- **Scaling:** StandardScaler för numeriska features
- **Feature engineering:** Age, Miles_Per_Year
- **Prestanda:** RMSE ~$481

### Pipeline

- **En klass:** `CarPipeline` hanterar allt
- **En fil:** `car_pipeline.pkl` innehåller allt
- **Korrekt:** OneHotEncoder istället för LabelEncoder

##  Resultat

- **RMSE:** ~$481 (genomsnittligt fel)
- **Användarvänlighet:** Maximal
- **Kodkvalitet:** Enkel och tydlig
- **Prestanda:** Snabb och effektiv

##  Mål uppfyllt

 **Enkel och koncis** - Minimal kod  
 **Användarvänlig** - Tydlig UI/UX  
 **Korrekt ML** - OneHotEncoder  
 **Endast RMSE** - Fokuserad bedömning  
 **En fil** - Allt i car_pipeline.pkl  
 **Snabb** - Optimal prestanda
