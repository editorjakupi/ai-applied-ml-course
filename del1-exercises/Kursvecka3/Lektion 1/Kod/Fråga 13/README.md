# MNIST Siffra Klassificering - Fråga 13 (FIXED VERSION)

##  Översikt

Denna lösning innehåller en komplett MNIST-klassificeringspipeline med en interaktiv Streamlit-app som predikerar handskrivna siffror med hög accuracy. **Alla preprocessing-problem är fixade!**

##  Filer

- **`Kodexempel_1_MNIST_Klassificering.ipynb`** - Huvudnotebook med MNIST-klassificering och modellträning
- **`mnist_streamlit_app.py`** - Interaktiv Streamlit-app för siffra-klassificering
- **`README.md`** - Denna fil

##  Snabbstart

### Steg 1: Installera dependencies

```bash
pip install streamlit numpy pandas matplotlib opencv-python pillow scikit-learn seaborn joblib
```

### Steg 2: Kör Streamlit-appen

```bash
streamlit run mnist_streamlit_app.py
```

### Steg 3: Träna modellen

1. Öppna appen i webbläsaren
2. Klicka på " Träna Modell" (tar några minuter)
3. Vänta tills modellen är tränad

### Steg 4: Testa klassificering

- **Ladda upp bild**: Välj en bildfil med en siffra
- **Testa med MNIST**: Använd inbyggda MNIST-bilder för testning

##  Modellprestanda

- **Modelltyp**: Extra Trees Classifier (förbättrad)
- **Test Accuracy**: ~97%+ (förbättrad från 95%)
- **Dataset**: 20,000 MNIST-bilder (2x mer data)
- **Träningstid**: ~5 minuter
- **Preprocessing**: StandardScaler (korrekt synkroniserad)

##  Teknisk implementation

### Preprocessing (FIXED!)

- **MNIST-bilder**: StandardScaler (mean=0, std=1) - samma som träning
- **Egna bilder**: Resize, färginvertering, kontrastförbättring, thresholding + StandardScaler
- **KRITISKT**: Alla bilder använder samma StandardScaler som modellen tränades på

### Modellträning

1. Laddar MNIST-datasetet (20,000 bilder)
2. Uppdelar i träning/validering/test (60/20/20%)
3. Skalar data med StandardScaler
4. Tränar Extra Trees modell
5. Sparar modell och metadata

### Prediktion

- Preprocessar input-bild
- Använder tränad modell för klassificering
- Visar sannolikheter för alla siffror (0-9)

##  Funktioner

### Streamlit-app

- **Modellträning**: Träna modellen direkt i appen
- **Bilduppladdning**: Ladda upp egna bilder
- **MNIST-test**: Testa med inbyggda MNIST-bilder
- **Visualisering**: Se original- och förbearbetade bilder
- **Sannolikheter**: Visa sannolikheter för alla siffror
- **Modellinfo**: Visa detaljerad modellinformation

### Jupyter Notebook

- **Komplett pipeline**: Data loading  träning  utvärdering
- **Modelljämförelse**: Jämför olika modeller
- **Feature importance**: Visa viktiga pixlar
- **Detaljerad analys**: Confusion matrix, classification report

##  Tips för bästa resultat

### För egna bilder:

- Använd svart text på vit bakgrund
- Rita siffran i mitten av bilden
- Gör siffran tydlig och stor
- Undvik dekorationer
- En siffra per bild

### För Paint-bilder:

- Använd svart färg för siffran
- Vit bakgrund
- Tydlig kontrast
- Ingen antialiasing

##  Felsökning

### Modellen predikerar fel:

1. Kontrollera att bilden har god kvalitet
2. Se till att siffran är i mitten
3. Använd svart text på vit bakgrund
4. Undvik dekorationer

### Appen startar inte:

1. Kontrollera att alla dependencies är installerade
2. Kör `streamlit run mnist_streamlit_app.py` från rätt mapp
3. Kontrollera att port 8501 är ledig

### Låg sannolikhet:

- Modellen är osäker på prediktionen
- Försök med en tydligare bild
- Kontrollera att preprocessing fungerar korrekt

##  Förbättringsmöjligheter

- **Fler modeller**: Lägg till Neural Networks
- **Data augmentation**: Öka träningsdata
- **Hyperparameter tuning**: Optimera modellparametrar
- **Rita-funktion**: Lägg till canvas för att rita siffror
- **Batch processing**: Klassificera flera bilder samtidigt

##  Lärdomar

Denna lösning demonstrerar:

- **Machine Learning pipeline**: Data  träning  utvärdering  produktion
- **Preprocessing**: Viktigheten av korrekt dataförbearbetning
- **Modelljämförelse**: Jämföra olika algoritmer
- **Streamlit**: Skapa interaktiva ML-appar
- **Model persistence**: Spara och ladda tränade modeller

##  Baserad på

- **Kodexempel 1**: MNIST Klassificering från kursen
- **Extra Trees**: Bästa modellen från notebooken
- **Scikit-learn**: Machine learning bibliotek
- **Streamlit**: Web app framework
