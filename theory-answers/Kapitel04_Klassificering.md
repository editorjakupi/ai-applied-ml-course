# Svar på Faktafrågor och Resonemangsfrågor - Kapitel 4: Klassificering

## Faktafrågor

### 1. Vad kännetecknar klassificeringsproblem? Ge några exempel på tillämpningsområden.

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

### 2. Förklara hur OvR- och OvO-algoritmerna fungerar.

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

### 3. Förklara följande utvärderingsmått:

#### a) Confusion Matrix

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

#### b) Accuracy

**Formel:** (TP + TN) / (TP + TN + FP + FN)

Andel korrekta prediktioner av alla prediktioner.

#### c) Precision

**Formel:** TP / (TP + FP)

"När modellen säger 'positiv', hur ofta har den rätt?"

#### d) Recall

**Formel:** TP / (TP + FN)

"Av alla som var positiva, hur många hittade modellen?"

Recall mäter andelen **positiva exempel** som modellen lyckades identifiera korrekt.

#### e) F1-Score

**Formel:** 2 × (Precision × Recall) / (Precision + Recall)

Harmoniskt medelvärde som balanserar precision och recall.

#### f) ROC-kurvan

- **X-axel:** False Positive Rate (FP / (FP + TN))
- **Y-axel:** True Positive Rate (TP / (TP + FN))
- Visar trade-off mellan TPR och FPR vid olika tröskelvärden
- AUC (Area Under Curve) ger ett mått på modellens prestanda

### 4. Vad är precision-recall tradeoff för något?

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

### 5. Några vanligt förekommande modeller för klassificeringsproblem är enligt nedan. Förklara översiktligt hur respektive modell fungerar.

#### a) Logistisk Regression

- **Grundprinciper:** Linjär modell med sigmoid-funktion
- **Output:** Sannolikhet mellan 0 och 1
- **Fördelar:** Tolkbar, snabb, bra baseline
- **Nackdelar:** Antar linjäritet

#### b) Support Vector Machines (SVM)

- **Grundprinciper:** Hittar optimalt separerande hyperplan
- **Kernel-trick:** Mappar data till högre dimension
- **Fördelar:** Effektiv i höga dimensioner, robust
- **Nackdelar:** Känslig för skalning

#### c) Beslutsträd

- **Grundprinciper:** Hierarkisk struktur med noder och kanter
- **Träning:** Väljer bästa feature att dela på (Information Gain)
- **Fördelar:** Mycket tolkbar, hanterar olika datatyper
- **Nackdelar:** Instabil, känslig för overfitting

#### d) Ensemble Learning

- **Grundprinciper:** Kombinerar flera modeller
- **Bagging:** Bootstrap Aggregating (Random Forest)
- **Boosting:** Sekventiell träning (AdaBoost, XGBoost)
- **Voting:** Majoritetsröstning eller vägt genomsnitt

#### e) Random Forest

- **Grundprinciper:** Ensemble av besluts-träd med bagging
- **Feature subsampling:** Väljer slumpmässigt subset av features
- **Fördelar:** Robust, ger feature importance
- **Nackdelar:** Mindre tolkbar än enskilda träd

#### f) Extra Trees

- **Grundprinciper:** Liknande Random Forest men med mer randomisering
- **Skillnad:** Väljer tröskelvärden slumpmässigt istället för optimalt
- **Fördelar:** Snabbare träning, mindre overfitting
- **Nackdelar:** Kan ha lägre prestanda

### 6. Vad innebär det att vi kan kolla på feature importance med hjälp av trädmodeller såsom beslutsträd eller random forest?

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

## Resonemangsfrågor

### 7. Stina säger till Kalle på lunchsamtalet "jag vill ha högsta möjliga precision för vår klassificeringsmodell". Kalle funderar ett tag och säger "men vad händer då med recall"? Vad hade du svarat? I vilka fall kan man tänka sig vilja ha en så hög precision som möjligt? I vilka fall kan det vara dåligt? Om vi tänker oss rättsväsendet där en slutgiltig dom kan leda till fängelse, vad kan vi då säga om precision-recall tradeoff?

**Kalles oro är berättigad:**
När vi ökar precision (genom att höja tröskelvärdet) så får vi ofta lägre recall. Detta är precision-recall tradeoff.

**När hög precision är bra:**

- **Spam-filter:** Du vill inte missa viktig e-post
- **Kvalitetskontroll:** Du vill inte kassera bra produkter
- **Rekommendationssystem:** Du vill bara rekommendera relevanta produkter
- **Medicinsk screening:** Du vill inte ge behandling till friska personer

**När hög precision kan vara dåligt:**

- **Cancer-screening:** Du missar många cancerfall (låg recall)
- **Säkerhet:** Du missar säkerhetsrisker
- **Kreditrisk:** Du nekar lån till många som skulle betalat tillbaka

**Rättsväsendet - en känslig balans:**
I rättsväsendet är både precision och recall kritiska:

- **Hög precision:** Vi vill inte döma oskyldiga (false positive)
- **Hög recall:** Vi vill inte låta skyldiga gå fria (false negative)

**Praktisk lösning:**

- Använd F1-score som balanserar båda
- Justera tröskelvärde baserat på samhällskostnad
- Implementera flera steg i processen

### 8. Förklara hur man kan tolka figur 4.8 på sidan 175.

**Figur 4.8 visar beslutsgränser för en logistisk regressionsmodell:**

**Vad grafen visar:**

- **X-axel:** Variabel X1 (ranges från -2.5 till 3.5)
- **Y-axel:** Variabel X2 (ranges från -2.5 till 2.5)
- **Data-punkter:** Två klasser (klass 0 = lila/svarta cirklar, klass 1 = gula cirklar)

**Bakgrundsfärgerna (sannolikhetskarta):**

- **Mörk lila/blå:** Låg sannolikhet för klass 1 (nära 0.0)
- **Ljus gul:** Hög sannolikhet för klass 1 (nära 1.0)
- **Grön/blågrön övergång:** Mellanliggande sannolikheter (0.2-0.8)

**Beslutsgränsen:**

- **Tröskelvärde 0.5:** Gränsen mellan klass 0 och klass 1
- **Diagonal linje:** Ungefär där sannolikheten är 0.5
- **Punkter på vänster sida:** Predikteras som klass 0
- **Punkter på höger sida:** Predikteras som klass 1

**Tolkning:**

- **Klass 0** (lila): Koncentrerade i övre vänstra området (låga X1, höga X2)
- **Klass 1** (gul): Koncentrerade i nedre högra området (höga X1, låga X2)
- **Överlappning:** Mitt i grafen finns det en övergångszon med osäker klassificering

**Vad detta betyder:**

- Modellen skapar en **probabilistisk separation** mellan klasserna
- Beslutsgränsen är **mjuk** (gradvis övergång) inte skarp
- Modellen är **osäker** i överlappningszonen
- **X1 och X2** tillsammans avgör klassificeringen

**Praktisk betydelse:**

- Visar hur modellen "ser" data
- Hjälper förstå varför vissa punkter klassificeras som de gör
- Visar modellens osäkerhet i gränsområdena

### 9. På sidan 209 står det "på träningsdatan använder vi `.fit_transform()`, på valideringsdatan och testdatan använder vi endast `.transform()`." Förklara logiken bakom detta.

**Grundprinciper:**

- **`.fit()`:** Lär sig parametrar från data (t.ex. medelvärde, standardavvikelse)
- **`.transform()`:** Applicerar de inlärda parametrarna på ny data
- **`.fit_transform()`:** Kombinerar båda stegen

**Varför denna separation:**

**1. Undvika Data Leakage:**

- Om vi använder `.fit_transform()` på valideringsdata lär modellen information från framtida data
- Detta ger optimistiska och missvisande resultat
- Vi simulerar inte verklig användning där vi bara har träningsdata

**2. Konsistent Transformation:**

- Samma parametrar (t.ex. samma medelvärde för skalning) används för alla datasets
- Validerings- och testdata transformeras på samma sätt som träningsdata
- Detta säkerställer att alla data behandlas likadant

**3. Simulera Produktionsmiljö:**

- I produktion har vi bara träningsdata att lära från
- Nya data måste transformeras med samma parametrar
- Vi kan inte "lära" från nya data

**Exempel med StandardScaler:**

```python
# Träning
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Lär sig mean, std och transformera

# Validering/Test
X_val_scaled = scaler.transform(X_val)  # Använd samma mean, std
X_test_scaled = scaler.transform(X_test)  # Använd samma mean, std
```

**Konsekvenser av fel:**

- **Optimistisk prestanda:** Modellen verkar bättre än den är
- **Dålig generalisering:** Modellen fungerar sämre på riktig data
- **Missvisande utvärdering:** Felaktiga beslut baserat på felaktiga resultat
