# Kapitel 4 - Klassificering: Faktafrågor

## Fråga 1: Grundläggande om Klassificering

**Vad kännetecknar klassificeringsproblem? Ge några exempel på tillämpningsområden.**

### Svar:

Klassificeringsproblem kännetecknas av:

- **Diskret output:** Modellen försöker förutsäga vilken kategori/klass en observation tillhör
- **Finit antal klasser:** Begränsat antal möjliga utfall
- **Kvalitativ prediktion:** Resultatet är en kategori, inte ett numeriskt värde

**Exempel på tillämpningsområden:**

- **Spam-detektering:** Klassificera e-post som spam eller icke-spam
- **Medicinsk diagnostik:** Identifiera sjukdomar från röntgenbilder
- **Kreditrisk:** Bedöma om en person ska få lån eller inte
- **Bildklassificering:** Känna igen objekt i bilder (katt/hund)
- **Ansiktsigenkänning:** Identifiera personer från foton
- **Kundsegmentering:** Kategorisera kunder baserat på beteende

---

## Fråga 2: OvR och OvO Algoritmer

**Förklara hur OvR- och OvO-algoritmerna fungerar.**

### Svar:

**One-vs-Rest (OvR) - "En mot Alla":**

- Tränar en binär modell per klass
- Varje modell lär sig att skilja en klass från alla andra
- För k klasser tränas k modeller
- Prediktion: Välj klass med högst sannolikhet

**Exempel för 3 klasser (Äpple, Banan, Apelsin):**

- Modell 1: Äpple vs (Banan + Apelsin)
- Modell 2: Banan vs (Äpple + Apelsin)
- Modell 3: Apelsin vs (Äpple + Banan)

**One-vs-One (OvO) - "En mot En":**

- Tränar en binär modell för varje par av klasser
- För k klasser tränas k(k-1)/2 modeller
- Prediktion: Röstning mellan alla modeller

**Exempel för 3 klasser:**

- Modell 1: Äpple vs Banan
- Modell 2: Äpple vs Apelsin
- Modell 3: Banan vs Apelsin

---

## Fråga 3: Utvärderingsmått

**Förklara följande utvärderingsmått:**

### a) Confusion Matrix

En tabell som visar hur modellen presterade på varje klass:

```
                    Predikterad
                Positiv  Negativ
Faktisk Positiv   TP      FN
Faktisk Negativ   FP      TN
```

- **TP (True Positive):** Rätt identifierade positiva
- **TN (True Negative):** Rätt identifierade negativa
- **FP (False Positive):** Felaktigt flaggade som positiva
- **FN (False Negative):** Missade positiva

### b) Accuracy

**Formel:** (TP + TN) / (TP + TN + FP + FN)

Andel korrekta prediktioner av alla prediktioner.

### c) Precision

**Formel:** TP / (TP + FP)

"När modellen säger 'positiv', hur ofta har den rätt?"

### d) Recall

**Formel:** TP / (TP + FN)

"Av alla som var positiva, hur många hittade modellen?"

Recall mäter andelen **positiva exempel** som modellen lyckades identifiera korrekt.

### e) F1-Score

**Formel:** 2 × (Precision × Recall) / (Precision + Recall)

Harmoniskt medelvärde som balanserar precision och recall.

### f) ROC-kurvan

- **X-axel:** False Positive Rate (FP / (FP + TN))
- **Y-axel:** True Positive Rate (TP / (TP + FN))
- Visar trade-off mellan TPR och FPR vid olika tröskelvärden
- AUC (Area Under Curve) ger ett mått på modellens prestanda

---

## Fråga 4: Precision-Recall Tradeoff

**Vad är precision-recall tradeoff för något?**

### Svar:

Precision-recall tradeoff är det förhållande att:

- **Högre precision** ofta leder till **lägre recall**
- **Högre recall** ofta leder till **lägre precision**

**Exempel:**

- Om vi sänker tröskelvärdet för att flagga spam:
  - Vi hittar fler spam (högre recall)
  - Men vi flaggar också mer legitim e-post som spam (lägre precision)

**Praktiska implikationer:**

- **Spam-filter:** Högre precision (vill inte missa viktig e-post)
- **Cancer-screening:** Högre recall (vill inte missa någon cancer)

---

## Fråga 5: Vanliga Klassificeringsmodeller

### a) Logistisk Regression

- **Grundprinciper:** Linjär modell med sigmoid-funktion
- **Output:** Sannolikhet mellan 0 och 1
- **Fördelar:** Tolkbar, snabb, bra baseline
- **Nackdelar:** Antar linjäritet

### b) Support Vector Machines (SVM)

- **Grundprinciper:** Hittar optimalt separerande hyperplan
- **Kernel-trick:** Mappar data till högre dimension
- **Fördelar:** Effektiv i höga dimensioner, robust
- **Nackdelar:** Känslig för skalning

### c) Beslutsträd

- **Grundprinciper:** Hierarkisk struktur med noder och kanter
- **Träning:** Väljer bästa feature att dela på (Information Gain)
- **Fördelar:** Mycket tolkbar, hanterar olika datatyper
- **Nackdelar:** Instabil, känslig för overfitting

### d) Ensemble Learning

- **Grundprinciper:** Kombinerar flera modeller
- **Bagging:** Bootstrap Aggregating (Random Forest)
- **Boosting:** Sekventiell träning (AdaBoost, XGBoost)
- **Voting:** Majoritetsröstning eller vägt genomsnitt

### e) Random Forest

- **Grundprinciper:** Ensemble av besluts-träd med bagging
- **Feature subsampling:** Väljer slumpmässigt subset av features
- **Fördelar:** Robust, ger feature importance
- **Nackdelar:** Mindre tolkbar än enskilda träd

### f) Extra Trees

- **Grundprinciper:** Liknande Random Forest men med mer randomisering
- **Skillnad:** Väljer tröskelvärden slumpmässigt istället för optimalt
- **Fördelar:** Snabbare träning, mindre overfitting
- **Nackdelar:** Kan ha lägre prestanda

---

## Fråga 6: Feature Importance

**Vad innebär det att vi kan kolla på feature importance med hjälp av trädmodeller såsom beslutsträd eller random forest?**

### Svar:

Feature importance visar hur viktig varje feature är för modellens prediktioner:

**Beräkningsmetoder:**

- **Information Gain:** Hur mycket information varje feature bidrar med
- **Gini Importance:** Baserat på Gini-index för varje feature
- **Permutation Importance:** Hur mycket prestandan sjunker när feature-värden blandas om

**Praktisk användning:**

- **Feature selection:** Ta bort irrelevanta features
- **Tolkbarhet:** Förstå vad som påverkar prediktionerna
- **Domain knowledge:** Validera mot expertis
- **Kostnadsbesparing:** Fokusera på viktiga features

**Exempel:**
I en modell för kreditrisk kan feature importance visa att:

- Inkomst är den viktigaste featuren (40%)
- Kredithistorik är näst viktigast (30%)
- Ålder har mindre betydelse (10%)
