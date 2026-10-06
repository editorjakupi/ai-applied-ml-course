# Svar på Faktafrågor och Resonemangsfrågor - Kapitel 1: Introduktion till maskininlärning

## Faktafrågor

### 1. Hur hänger AI, ML och DL ihop?

AI (Artificiell Intelligens), ML (Maskininlärning) och DL (Djupinlärning) hänger ihop som en hierarki:

- **AI** är det övergripande fältet som handlar om att skapa intelligenta maskiner
- **ML** är en delmängd av AI som fokuserar på algoritmer som kan lära från data
- **DL** är en delmängd av ML som använder neurala nätverk med många lager

**Förhållandet:** AI ⊃ ML ⊃ DL

### 2. Vilka är de fyra problemkategorierna inom ML?

De fyra problemkategorierna inom maskininlärning är:

1. **Regression** - Prediktera kontinuerliga värden (t.ex. huspriser, lön)
2. **Klassificering** - Prediktera diskreta kategorier (t.ex. spam/ej spam, kundsegment)
3. **Dimensionsreducering** - Minska antalet variabler (t.ex. PCA)
4. **Klustring** - Gruppera liknande data utan fördefinierade kategorier

### 3. Förklara följande koncept

#### a) Syftet med att dela upp data i träningsdata, valideringsdata och testdata

- **Träningsdata**: Används för att lära modellen samband mellan features och target
- **Valideringsdata**: Används för att utvärdera modeller under utveckling och välja hyperparametrar
- **Testdata**: Används för slutlig, opartisk utvärdering av den valda modellen

**Syfte**: Förhindra överanpassning och få en realistisk bedömning av modellens prestanda på ny data.

#### b) K-delad korsvalidering

K-delad korsvalidering är en teknik där data delas upp i K lika stora delar (folds). Modellen tränas K gånger, varje gång med K-1 folds som träning och 1 fold som validering. Varje fold används exakt en gång som validering.

**Fördelar**:
- Mer robust utvärdering
- Bättre användning av begränsad data
- Mindre beroende av en specifik datauppdelning

#### c) Root Mean Squared Error (RMSE)

RMSE är ett mått för att utvärdera regressionsmodeller. Det beräknas som kvadratroten av medelvärdet av de kvadrerade felen:

RMSE = √(1/n × Σ(y_i - ŷ_i)²)

**Egenskaper**:
- Samma enhet som target-variabeln
- Straffar stora fel mer än små fel (på grund av kvadrering)
- Lägre RMSE = bättre modell

#### d) Hyperparametrar vs Parametrar

- **Hyperparametrar**: Konfigureras före träning ("hur vi lär oss")
  - Exempel: antal träd i Random Forest, learning rate, antal lager i neuralt nätverk
  - Väljs av data scientist

- **Parametrar**: Lärs av modellen under träning ("vad vi lär oss")
  - Exempel: koefficienter i linjär regression, vikter i neuralt nätverk
  - Optimeras automatiskt av algoritmen

#### e) Grid Search

Grid Search är en metod för hyperparameteroptimering som systematiskt testar alla kombinationer av hyperparametrar i ett fördefinierat rutnät (grid).

**Namnförklaring**:
- **"Grid"**: Hyperparametrarna bildar ett rutnät av kombinationer
- **"Search"**: Algoritmen söker genom alla kombinationer för att hitta den bästa

**refit=True**: När GridSearchCV hittar bästa hyperparametrarna, tränar den om modellen på hela träningsdatan med dessa parametrar.

#### f) Kategorisk data och hantering

**Kategorisk data** är data som representerar kategorier eller grupper.

**Typer**:
- **Nominal data**: Kategorier utan ordning (t.ex. färg: röd, blå, grön)
- **Ordinal data**: Kategorier med ordning (t.ex. utbildningsnivå: grundskola, gymnasium, universitet)

**Hanteringsmetoder**:
- **One-hot encoding**: Skapar binära kolumner för varje kategori
- **Dummy-variable encoding**: Som one-hot men droppar en kategori för att undvika multicollinearity
- **Ordinal encoding**: Mappar ordinala kategorier till numeriska värden

#### g) Feature Engineering

Feature Engineering är processen att skapa, välja och transformera variabler (features) för att förbättra modellens prestanda.

**Exempel**:
- Skapa nya features från befintliga (t.ex. ålder från födelsedatum)
- Kombinera features (t.ex. pris per kvadratmeter)
- Transformera features (t.ex. logaritmisk transformation)
- Hantera saknade värden

#### h) Principle of Parsimony

Principle of Parsimony (Ockhams rakkniv) säger att vi ska föredra enklare modeller framför komplexa modeller när de presterar lika bra.

**Fördelar med enklare modeller**:
- Lättare att förstå och tolka
- Mindre risk för överanpassning
- Snabbare att träna och använda
- Mer robusta

### 4. Vad menas med att "en modell är en förenkling av verkligheten"?

En modell är en förenkling eftersom den:

- **Väljer ut viktiga faktorer** och ignorerar mindre viktiga
- **Gör antaganden** om samband mellan variabler
- **Idealiserar verkligheten** för att göra den hanterbar
- **Fokuserar på specifika aspekter** av problemet

**Exempel**: En linjär modell antar linjära samband, men verkligheten är ofta mer komplex.

### 5. Vad menas med att en modell är "överanpassad" eller overfitted?

En överanpassad modell har lärt sig träningsdatan för väl, inklusive brus och slumpmässiga variationer. Den presterar bra på träningsdata men dåligt på ny data.

**Symptom**:
- Hög precision på träningsdata
- Låg precision på validering/testdata
- Komplex modell som fångar brus

**Lösningar**:
- Fler träningsdata
- Regularisering
- Enklare modellarkitektur

### 6. Högre är bättre i scikit-learn scoring, vad innebär det?

Scikit-learn förväntar sig att högre värden betyder bättre prestanda. För felmått som RMSE (där lägre är bättre) används därför negativa värden.

**Exempel**:
- `neg_mean_squared_error`: Negativ MSE
- `neg_root_mean_squared_error`: Negativ RMSE

**Logik**: -2.5 är "högre" än -3.0, så modellen med -2.5 presterar bättre.

### 7. Vad är tvärsnittsdata, tidsseriedata och paneldata?

**Tvärsnittsdata**: Observationer för olika individer vid en tidpunkt
- Exempel: Inkomst för olika personer år 2023

**Tidsseriedata**: Observationer över tid för en individ
- Exempel: Aktiepris för ett företag över 5 år

**Paneldata**: Observationer för olika individer över tid
- Exempel: Inkomst för olika personer över 10 år

## Resonemangsfrågor

### 8. Ge exempel på verkliga tillämpningsområden inom ML

**Rekommendationssystem**:
- Netflix, Spotify, Amazon
- Predikterar användarpreferenser

**Bildklassificering**:
- Medicinsk diagnostik
- Självkörande bilar
- Säkerhetskameror

**Naturligt språk**:
- Chatbots (ChatGPT)
- Översättning
- Sentimentanalys

**Finans**:
- Kreditbedömning
- Aktieprediktion
- Bedrägeridetektering

**E-handel**:
- Prisoptimering
- Lagerhantering
- Kundsegmentering

### 9. Förklara logiken bakom negativ mean squared error

**Problemet**:
- MSE är ett felmått (lägre = bättre)
- Scikit-learn förväntar sig "högre = bättre"

**Lösningen**:
- Använd negativ MSE: `-MSE`
- Nu gäller: högre (mindre negativt) = bättre

**Exempel**:
- Modell A: MSE = 2.5  Score = -2.5
- Modell B: MSE = 3.0  Score = -3.0
- Modell A är bättre (-2.5 > -3.0)

**Fördelar**:
- Konsekvent med scikit-learns konventioner
- Enkel jämförelse mellan modeller
- Fungerar med alla optimeringsalgoritmer
