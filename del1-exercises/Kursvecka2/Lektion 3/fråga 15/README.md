# Fråga 15: Diamantprisprediktion 

## Beskrivning

Komplett ML-lösning för att prediktera diamantpriser baserat på diamantens egenskaper med professionell Pipeline-struktur och förbättrad feature engineering.

## Filer i mappen

###  Dataset

- **`diamonds.csv`** - Dataset med 53,864 diamanter och deras egenskaper

###  ML-analys

- **`Fråga_15_Diamantpriser.ipynb`** - Jupyter notebook med komplett ML-flöde

###  Tränad pipeline

- **`diamond_pipeline.pkl`** - Komplett förbättrad pipeline (preprocessor + modell i en fil)

###  Applikation

- **`diamond_price_app.py`** - Streamlit-app för interaktiv prediktion

## Så här använder du systemet

### 1. Kör Jupyter notebook för ML-analys

```bash
jupyter notebook Fråga_15_Diamantpriser.ipynb
```

- Kör alla celler för att träna pipeline
- Pipeline sparas automatiskt som `diamond_pipeline.pkl`

### 2. Kör Streamlit-appen

```bash
streamlit run diamond_price_app.py
```

## ML-flöde i notebooken

### 7 steg enligt bokens metodik:

1. **Problemdefinition** - Förstå vad vi ska lösa
2. **Dataåtkomst** - Ladda och inspektera data
3. **EDA** - Utforska och visualisera data
4. **Förbehandling** - Rensa och förbered data
5. **Modellering** - Träna och utvärdera modeller
6. **Presentation** - Visa resultat
7. **Produktionssättning** - Spara modell för användning

## Pipeline-struktur

### Professionell ML-pipeline:

```
Input Data  Feature Engineering  ColumnTransformer  Model  Prediction
```

### Pipeline-komponenter:

1. **Feature Engineering** - Skapar förbättrade features
2. **ColumnTransformer** - Hanterar numeriska och kategoriska features
3. **Model** - Random Forest eller Linear Regression

### Fördelar med Pipeline:

- **En fil** istället för två separata
- **Automatisk preprocessing** - inget manuellt arbete
- **Konsistent** - samma preprocessing vid träning och prediktion
- **Professionell** - industri-standard för ML-produktion

## Features

### Grundläggande diamantegenskaper

- **Carat** - Vikt i karat
- **Cut** - Skärning (Ideal, Premium, Good, etc.)
- **Color** - Färg (D, E, F, G, H, I, J)
- **Clarity** - Klarhet (IF, VVS1, VVS2, VS1, VS2, SI1, SI2)
- **Depth** - Djup (%)
- **Table** - Bord (%)
- **Dimensions** - Längd (x), bredd (y), höjd (z)

### Förbättrade features (Feature Engineering)

- **Volume:** `x * y * z` (diamantens volym)
- **Table/Depth Ratio:** `table / depth` (proportioner)
- **Carat/Volume:** `carat / volume` (densitet)
- **Total Depth:** `depth * table / 100` (kombinerad djup)

### Modellprestanda

- **RMSE:** $898 (mycket lågt fel!)
- **Genomsnittligt fel:** $898 per prediktion
- **Modell:** Förbättrad Random Forest Pipeline
- **Förbättring:** $250 bättre än grundmodellen
- **Körningstid:** ~10 sekunder (optimerad för snabbhet)

### Feature Engineering

- **Volume:** Diamantens volym (x × y × z)
- **Table/Depth Ratio:** Proportioner för skärning
- **Carat/Volume:** Densitet av diamanten
- **Total Depth:** Kombinerad djupmätning
- **StandardScaler:** Normaliserar numeriska features
- **OneHotEncoder:** Encodar kategoriska variabler

## Användningsområden

-  Juvelexperter för prissättning
-  Försäkringsbolag för värdering
-  Auktionshus för utgångspriser
-  Köpare/säljare för marknadspriser

## Teknisk stack

- **Python** - Programmeringsspråk
- **Jupyter Notebook** - ML-analys och visualisering
- **scikit-learn** - Machine Learning och Pipeline
- **pandas** - Datahantering
- **Streamlit** - Web-applikation
- **matplotlib/seaborn** - Visualisering

---

_Byggd med  för ML-utbildning_
