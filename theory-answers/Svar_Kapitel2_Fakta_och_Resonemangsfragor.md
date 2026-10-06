# Svar på Faktafrågor och Resonemangsfrågor - Kapitel 2: Ett ML projekt från början till slut

## Faktafrågor

### 1. I kapitlet beskrivs en checklista med sju steg. Beskriv de sju stegen översiktligt. I verkligheten, följs dessa steg i en rak progression eller arbetar man generellt sett mer iterativt?

**De sju stegen:**

1. **Definiera problemet** - Vad vill vi lösa?
2. **Få tillgång till data** - Samla in relevant data
3. **Exploratory Data Analysis (EDA)** - Förstå datan
4. **Bearbeta datan** - Rensa och förbereda data
5. **ML-modellering** - Träna och utvärdera modeller
6. **Spara modellen** - För framtida användning
7. **Produktionssättning** - Använd modellen i verkligheten

**Iterativt arbete:**

- **Nej, man följer inte stegen i rak progression**
- **Iterativt arbete** är normen i verkligheten
- Man går tillbaka och justerar baserat på resultat
- EDA kan leda till att man behöver mer data (steg 2)
- Modellering kan visa att man behöver bättre dataförbehandling (steg 4)
- **Agile-approach** med snabba iterationer

### 2. Vad menas med att en modell produktionssätts?

**Produktionssättning** betyder att modellen används i verkligheten för att göra prediktioner på ny data.

**Vad det innebär:**

- **Modellen är live** och används av slutanvändare
- **Automatiserad prediktion** på ny data
- **Integration** med befintliga system
- **Skalbarhet** för att hantera många användare
- **Övervakning** av modellens prestanda

**Exempel:**

- **Netflix**: Rekommendationssystem som fungerar för alla användare
- **Bank**: Kreditbedömning som används för låneansökningar
- **E-handel**: Prisoptimering som fungerar i realtid

**Utmaningar:**

- **Data drift** - nya data kan skilja sig från träningsdata
- **Prestanda** - modellen måste vara snabb nog
- **Tillförlitlighet** - systemet måste vara stabilt
- **Underhåll** - modeller behöver uppdateras regelbundet

### 3. Vad är scikit-learn för något? Biblioteket följer några centrala designprinciper. Vilka är dessa? Vad är estimators, predictors och transformers?

**Vad är scikit-learn?**

- **Python-bibliotek** för maskininlärning
- **Open source** och gratis att använda
- **Omfattande** samling av ML-algoritmer
- **Användarvänligt** och väl dokumenterat

**Centrala designprinciper:**

1. **Konsistens** - Alla modeller följer samma interface
2. **Inspection** - Enkel tillgång till modellparametrar
3. **Non-proliferation of classes** - Få, väl definierade klasser
4. **Composition** - Modeller kan kombineras
5. **Sensible defaults** - Bra standardvärden

**Estimators, Predictors och Transformers:**

**Estimators**:
- Objekt som kan lära sig från data
- Har `.fit()` metod
- Exempel: `LinearRegression()`, `RandomForestRegressor()`

**Predictors**:
- Estimators som kan göra prediktioner
- Har `.predict()` metod
- Exempel: Alla regressions- och klassificeringsmodeller

**Transformers**:
- Objekt som kan transformera data
- Har `.fit()` och `.transform()` metoder
- Exempel: `StandardScaler()`, `OneHotEncoder()`

### 4. Vad är TensorFlow och Keras?

**TensorFlow:**

- **Open source bibliotek** för maskininlärning
- **Utvecklat av Google**
- **Fokuserar på deep learning** och neurala nätverk
- **Stödjer både CPU och GPU**
- **Används för storskaliga ML-system**

**Keras:**

- **High-level API** för neurala nätverk
- **Bygger på TensorFlow** (eller andra backends)
- **Användarvänligt** och intuitivt
- **Snabb prototypning** av neurala nätverk
- **Modulär design** - enkelt att bygga komplexa modeller

**Samband:**

- **Keras är nu en del av TensorFlow**
- **TensorFlow.keras** är den officiella implementationen
- **Keras ger enklare interface** till TensorFlow's kraftfulla funktioner
- **TensorFlow hanterar low-level operationer** (GPU-optimering, etc.)

## Resonemangsfrågor

### 5. Kalle och Stina diskuterar maskininlärning över en lunch. Kalle säger "om jag tränat en modell och den inte presterar bra nog på testdatan så justerar jag den tills den gör det." Stina säger "det är ett stort fel att göra så, det enda du då åstadkommer är att du överanpassar testdatan. Hela syftet med testdatan försvinner då". Vad säger du om deras dialog?

**Stina har helt rätt!**

**Vad Kalle gör fel:**

- **Justerar modellen baserat på testdata**
- **Överanpassar testdatan** (overfitting)
- **Förstör syftet med testdata**
- **Får optimistiska resultat** som inte representerar verklig prestanda

**Varför testdata är viktig:**

- **Oberoende utvärdering** av modellens prestanda
- **Simulerar verklig användning** på ny data
- **Hjälper att upptäcka overfitting**
- **Ger en ärlig bedömning** av modellens kvalitet

**Rätt process:**

1. **Träna modell** på träningsdata
2. **Justera hyperparametrar** baserat på valideringsdata
3. **Utvärdera EN gång** på testdata
4. **Testdata används aldrig för justering**

**Kalles misstag:**

- **Data leakage** - information från testdata läcker in i modellen
- **Optimistisk bias** - modellen presterar bättre än den egentligen gör
- **Förlorad tillförlitlighet** - ingen ärlig bedömning av prestanda

### 6. Många AI/ML projekt uppnår inte de ursprungligen satta målen eller att ens passera någon form av prototyp-stadie. Vad tror du detta beror på och hur ska vi förhålla oss till det?

**Vanliga orsaker till misslyckande:**

**1. Orealistiska förväntningar:**

- **"AI löser allt"** - för höga förväntningar
- **Underestimera komplexitet** - ML är inte magi
- **Orealistiska tidsramar** - ML-projekt tar tid

**2. Data-problem:**

- **Dålig datakvalitet** - skräp in, skräp ut
- **Otillräcklig data** - för lite träningsdata
- **Saknade värden** - ofullständig data
- **Data bias** - fördomar i datan

**3. Tekniska utmaningar:**

- **Komplexitet** - för avancerade modeller
- **Skalbarhet** - systemet kan inte hantera belastning
- **Integration** - svårt att integrera med befintliga system

**4. Organisatoriska problem:**

- **Lack of expertise** - ingen ML-kompetens
- **Motstånd till förändring** - organisationen är inte redo
- **Dålig kommunikation** - missförstånd mellan team

**Hur vi ska förhålla oss till det:**

- **Acceptera att misslyckanden händer** - det är normalt
- **Lär av misstagen** - varje misslyckande ger insikter
- **Starta smått** - prototyper innan storskaliga projekt
- **Sätt realistiska mål** - gradvis förbättring
- **Fokusera på värde** - lösa verkliga problem

### 8. Förklara vad koden nedan gör. Varför är det viktigt att kunna spara en modell?

**Vad koden gör:**

Koden demonstrerar hur man sparar och laddar en tränad modell:

1. **Skapar syntetisk data** med `make_regression()`
2. **Tränar en linjär regressionsmodell** på datan
3. **Sparar modellen** till fil med `dump()` från joblib
4. **Läser tillbaka modellen** från fil med `load()`
5. **Använder den laddade modellen** för att göra prediktioner

**Varför är det viktigt att spara modeller?**

1. **Vi vill inte träna om modellen varje gång** - träning kan ta lång tid
2. **Modeller kan vara dyra att träna** - kostar tid och resurser
3. **Vi vill kunna använda modellen i produktion** - modellen måste vara tillgänglig
4. **Vi vill dela modellen med andra** - kollegor eller system kan använda den
5. **Versionering** - spara olika versioner av modeller för jämförelse
6. **Reproducerbarhet** - samma modell kan användas flera gånger
7. **Backup** - säkerhetskopiera tränade modeller

**Praktiska fördelar:**

- **Snabbare deployment** - ingen behov av att träna om
- **Konsistens** - samma modell används överallt
- **Resurseffektivitet** - sparar beräkningskraft
- **Flexibilitet** - enkelt att byta mellan olika modeller
