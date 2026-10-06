# Kapitel 5: Dimensionsreduktion - Faktafrågor

## Fråga 1: Vad menas med curse of dimensionality?

**Svar:**
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

## Fråga 2: Vad är dimensionsreducering och varför görs det?

**Svar:**
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

## Fråga 3: Förklara översiktligt hur PCA fungerar. Använd figur 5.4 på sidan 224 i din förklaring.

**Svar:**
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

## Fråga 4: Hur kan kernel PCA utvärderas?

**Svar:**
Kernel PCA är en utvidgning av PCA som kan hantera icke-linjära relationer genom att använda kernel-trick.

**Utvärderingsmetoder:**

### 1. **Rekonstruktionsfel**

- Mät hur väl data kan rekonstrueras
- Mean Squared Error (MSE)
- Jämför med originaldata

### 2. **Variance Explained**

- Andel av total varians som bevaras
- Cumulative variance plot
- Välj antal komponenter baserat på önskad variansförklaring

### 3. **Cross-validation**

- Dela data i träning/validering
- Träna på träningsdata
- Utvärdera på valideringsdata
- Undvik överanpassning

### 4. **Downstream Task Performance**

- Använd reducerade features för klassificering/regression
- Jämför prestanda med originalfeatures
- Mät accuracy, precision, recall

### 5. **Visualisering**

- 2D/3D scatter plots
- Kvalitativ bedömning av kluster
- Separation mellan klasser

### 6. **Kernel Selection**

- Testa olika kernels (RBF, polynomial, sigmoid)
- Jämför prestanda
- Välj kernel baserat på dataegenskaper

**Praktiska tips:**

- Standardisera data före kernel PCA
- Välj kernel-parametrar noggrant
- Överväg beräkningskostnad
- Validera resultat på oberoende data
