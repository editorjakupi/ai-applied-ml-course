# Svar på Faktafrågor och Resonemangsfrågor - Kapitel 5: Dimensionsreduktion

## Faktafrågor

### 1. Vad menas med curse of dimensionality?

Curse of dimensionality refererar till de problem som uppstår när man arbetar med data i höga dimensioner:

**Huvudproblem:**

1. **Avståndsmått blir mindre meningsfulla**: I höga dimensioner blir alla punkter ungefär lika långt från varandra
2. **Ökad beräkningskomplexitet**: Algoritmer blir exponentiellt långsammare
3. **Överanpassning**: Modeller blir mer komplexa och riskerar att memorera träningsdata
4. **Sparsitet**: Data blir glesare i höga dimensioner, vilket gör det svårare att hitta mönster
5. **Visualisering**: Omöjligt att visualisera data i mer än 3 dimensioner

**Lösningar:**

- Dimensionsreduktion (PCA, t-SNE)
- Feature selection
- Regularisering
- Ensemble-metoder

### 2. Vad är dimensionsreducering och varför görs det?

Dimensionsreducering är processen att reducera antalet features (variabler) i en dataset samtidigt som man behåller så mycket viktig information som möjligt.

**Varför görs det:**

1. **Reducera beräkningskomplexitet**: Färre dimensioner = snabbare beräkningar
2. **Förbättra visualisering**: Möjliggör visualisering av högdimensionell data
3. **Minska överanpassning**: Färre parametrar = mindre risk för overfitting
4. **Ta bort brus och redundans**: Eliminera irrelevanta eller korrelerade features
5. **Förbättra modellprestanda**: Bättre generalisering på nya data
6. **Hantera curse of dimensionality**: Minska problem med höga dimensioner

**Metoder:**

- **Feature selection**: Väljer ut viktiga features
- **Feature extraction**: Skapar nya features från befintliga (PCA, t-SNE, LDA)

### 3. Förklara översiktligt hur PCA fungerar. Använd figur 5.4 på sidan 224 i din förklaring.

PCA (Principal Component Analysis) är en linjär dimensionsreduktionsteknik som hittar de riktningar (huvudkomponenter) där data varierar mest.

**Algoritm:**

1. **Standardisering**: Centrera och skala data
2. **Kovariansmatris**: Beräkna C = (1/n) X^T X
3. **Eigenvalues och Eigenvectors**: Hitta egenvärden och egenvektorer
4. **Projektion**: Projicera data på huvudkomponenter

**Figur 5.4 visar:**

- Originaldata i 2D
- Huvudkomponenter (PC1 och PC2) som riktningar med störst varians
- Projektion av data på huvudkomponenterna
- PC1 förklarar mer varians än PC2

**Matematisk grund:**

```
C = P Λ P^T
```

där P är matris med egenvektorer, Λ är diagonal matris med egenvärden.

**Fördelar:**

- Linjär transformation
- Bevarar global struktur
- Matematiskt välgrundad
- Snabb beräkning

**Nackdelar:**

- Antagande om linjära relationer
- Känslig för outliers
- Kan förlora viktig information

### 4. Hur kan kernel PCA utvärderas?

Kernel PCA är en utvidgning av PCA som kan hantera icke-linjära relationer genom att använda kernel-trick.

**Utvärderingsmetoder:**

#### 1. **Rekonstruktionsfel**

- Mät hur väl data kan rekonstrueras
- Mean Squared Error (MSE)
- Jämför med originaldata

#### 2. **Variance Explained**

- Andel av total varians som bevaras
- Cumulative variance plot
- Välj antal komponenter baserat på önskad variansförklaring

#### 3. **Cross-validation**

- Dela data i träning/validering
- Träna på träningsdata
- Utvärdera på valideringsdata
- Undvik överanpassning

#### 4. **Downstream Task Performance**

- Använd reducerade features för klassificering/regression
- Jämför prestanda med originalfeatures
- Mät accuracy, precision, recall

#### 5. **Visualisering**

- 2D/3D scatter plots
- Kvalitativ bedömning av kluster
- Separation mellan klasser

#### 6. **Kernel Selection**

- Testa olika kernels (RBF, polynomial, sigmoid)
- Jämför prestanda
- Välj kernel baserat på dataegenskaper

**Praktiska tips:**

- Standardisera data före kernel PCA
- Välj kernel-parametrar noggrant
- Överväg beräkningskostnad
- Validera resultat på oberoende data

## Resonemangsfrågor

### 5. Stina påstår att man i maskininlärning alltid vill ha modeller som genomför så bra prediktioner som möjligt. Kalle påstår att det inte riktigt stämmer eftersom tid också är en viktig aspekt. Både för själva modellträningen och för själva prediktionerna. Vad säger du?

Kalle har rätt - tid är en kritisk aspekt i maskininlärning som ofta överskuggas av fokus på prediktionsprestanda. Här är varför:

**Varför tid är viktigt:**

#### 1. **Modellträning**

- **Stora dataset**: Träning kan ta timmar eller dagar
- **Iterativ utveckling**: Snabbare träning = fler experiment
- **Resurskostnad**: Beräkningskraft kostar pengar
- **Deadlines**: Projekt har tidsramar

#### 2. **Prediktioner (Inference)**

- **Real-time applikationer**: Måste svara inom millisekunder
- **Skalbarhet**: Tusentals användare samtidigt
- **Resursbegränsningar**: Mobilappar, edge devices
- **Användarupplevelse**: Långsam respons = dålig UX

#### 3. **Praktiska överväganden**

- **Maintenance**: Komplexa modeller är svårare att underhålla
- **Deployment**: Enkla modeller är lättare att implementera
- **Debugging**: Snabbare att felsöka enkla modeller
- **Updates**: Lättare att uppdatera snabba modeller

**Balans mellan prestanda och tid:**

#### **När prestanda är viktigast:**

- Medicinska diagnoser
- Säkerhetskritiska system
- Offline-analys
- Batch-processing

#### **När tid är viktigast:**

- Real-time rekommendationer
- Mobilappar
- Trading-system
- Interaktiva applikationer

**Praktiska lösningar:**

1. **Model compression**: Kvantisering, pruning
2. **Ensemble simplification**: Färre modeller
3. **Feature selection**: Färre features
4. **Hardware optimization**: GPU, TPU
5. **Caching**: Spara vanliga prediktioner

**Slutsats**: Optimal balans beror på användningsfall. I verkligheten måste man ofta kompromissa mellan prestanda och hastighet.

### 6. Efter att vi genomfört en PCA, vad händer med tolkningen av variablerna?

Efter PCA förändras tolkningen av variablerna avsevärt, vilket är en av de största utmaningarna med metoden:

**Vad händer med tolkningen:**

#### 1. **Förlust av originalvariabler**

- **Originalvariabler**: Tydlig tolkning (ålder, inkomst, etc.)
- **Huvudkomponenter**: Abstrakta kombinationer av originalvariabler
- **Svårt att tolka**: Vad betyder "PC1" i praktiken?

#### 2. **Matematisk transformation**

- **PC1** = w1×variabel1 + w2×variabel2 + ... + wn×variabeln
- **Vikter (w)**: Visar bidrag från varje originalvariabel
- **Tolkning**: Kräver analys av vikterna

#### 3. **Praktiska utmaningar**

- **Affärsstakeholders**: Förstår inte "PC1"
- **Beslutsfattning**: Svårt att basera beslut på abstrakta komponenter
- **Kommunikation**: Måste översätta till förståeligt språk

**Sätt att hantera tolkning:**

#### 1. **Analysera komponentvikter**

```python
# Exempel: PC1 = 0.7×ålder + 0.5×inkomst - 0.3×utbildning
# Tolkning: PC1 representerar "mogenhet/erfarenhet"
```

#### 2. **Skapa meningsfulla namn**

- **PC1**: "Socioekonomisk status"
- **PC2**: "Livsstil"
- **PC3**: "Riskbenägenhet"

#### 3. **Visualisera bidrag**

- **Heatmaps**: Visa vikter för varje komponent
- **Bar charts**: Jämför vikter mellan komponenter
- **Biplots**: Kombinera data och variabler

#### 4. **Domänkunskap**

- **Expertis**: Använd domänkunskap för tolkning
- **Validering**: Kontrollera att tolkningar är rimliga
- **Iteration**: Justera baserat på feedback

**Alternativa metoder för bättre tolkning:**

#### 1. **Feature Selection**

- Behåll originalvariabler
- Bättre tolkning
- Kan förlora information

#### 2. **LDA (Linear Discriminant Analysis)**

- Övervakad metod
- Bättre tolkning
- Kräver klasslabels

#### 3. **Factor Analysis**

- Modellerar latenta faktorer
- Bättre tolkning
- Mer komplex

**Praktiska rekommendationer:**

1. **Dokumentera tolkningar**: Skriv ner vad varje komponent betyder
2. **Validera med experter**: Kontrollera att tolkningar är korrekta
3. **Använd visualisering**: Hjälp stakeholders att förstå
4. **Överväg alternativ**: Feature selection om tolkning är kritisk
5. **Kommunikera tydligt**: Förklara begränsningar och fördelar

**Slutsats**: PCA förbättrar prestanda men försämrar tolkning. Detta är en avvägning som måste göras baserat på projektets mål och stakeholders behov.
