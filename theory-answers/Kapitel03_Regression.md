# Svar på Faktafrågor och Resonemangsfrågor - Kapitel 3: Regression

## Faktafrågor

### 1. Vad kännetecknar regressionsproblem? Ge några exempel på tillämpningsområden.

**Regressionsproblem** kännetecknas av att vi försöker förutsäga en **kontinuerlig numerisk variabel** (target/output) baserat på en eller flera oberoende variabler (features/inputs).

**Huvudkännetecken:**

- **Kontinuerlig output**: Target-variabeln kan anta alla värden inom ett intervall
- **Numerisk prediktion**: Vi försöker förutsäga ett specifikt numeriskt värde
- **Kvantitativ analys**: Resultatet är ett måttbart värde

**Exempel på tillämpningsområden:**

1. **Ekonomi och Finans**
   - Lönsprediktion baserat på ålder, utbildning, erfarenhet
   - Aktieprisprediktion baserat på ekonomiska indikatorer
   - Försäljningsprognoser baserat på marknadsdata

2. **Fastigheter**
   - Husprisprediktion baserat på storlek, ålder, plats
   - Hyresprisprediktion baserat på läge och faciliteter

3. **Transport och Logistik**
   - Bilprisprediktion baserat på miltal, ålder, märke
   - Drivmedelsförbrukning baserat på körstil och väglag
   - Leveranstidsprediktion baserat på avstånd och trafik

4. **Medicin och Hälsa**
   - Diabetes progression baserat på blodvärden
   - Blodtrycksprediktion baserat på livsstil
   - Medicindosering baserat på patientdata

5. **Jordbruk och Miljö**
   - Skördeprediktion baserat på väder och gödning
   - Temperaturprediktion baserat på historisk data
   - Luftkvalitetsprediktion baserat på utsläpp

6. **Industri och Produktion**
   - Produktionsvolym baserat på resurser
   - Underhållsbehov baserat på maskindata
   - Kvalitetsmått baserat på processparametrar

**Viktigt att komma ihåg:** Regression handlar alltid om att förutsäga ett **kontinuerligt värde**, till skillnad från klassificering som förutsäger kategorier.

### 2. Förklara utvärderingsmåtten RMSE, MSE och MAE.

#### **RMSE (Root Mean Square Error)**

**Formel:** RMSE = √(1/n × Σ(y_true - y_pred)²)

**Vad det mäter:**

- Kvadratroten av genomsnittliga kvadratiska felet
- Ger fel i samma enhet som target-variabeln

**Fördelar:**

- Straffar stora fel hårdare (kvadrering)
- Samma enhet som target (tolkbart)
- Standardmått inom regression

**Exempel:** Om vi predikerar lön i kronor, blir RMSE också i kronor

#### **MSE (Mean Square Error)**

**Formel:** MSE = 1/n × Σ(y_true - y_pred)²

**Vad det mäter:**

- Genomsnittliga kvadratiska felet
- Samma som RMSE men utan kvadratrot

**Fördelar:**

- Billigare att beräkna än RMSE
- Samma rangordning som RMSE
- Matematiskt enklare att optimera

**Exempel:** Om RMSE = 1000 kr, så är MSE = 1,000,000 kr²

#### **MAE (Mean Absolute Error)**

**Formel:** MAE = 1/n × Σ|y_true - y_pred|

**Vad det mäter:**

- Genomsnittliga absoluta felet
- Alla fel värderas linjärt

**Fördelar:**

- Mindre känsligt för extremvärden
- Samma enhet som target
- Robust mot outliers

**Exempel:** Mindre känsligt för stora fel än RMSE

#### **Jämförelse RMSE vs MAE:**

**Exempel:**

- Modell 1: Fel på 10 och 10  RMSE = 10, MAE = 10
- Modell 2: Fel på 5 och 15  RMSE = 11.2, MAE = 10

**Slutsats:**

- RMSE föredrar Modell 1 (jämnare fel)
- MAE är neutral (samma totala fel)

**Praktisk regel:**

- Använd **RMSE** när stora fel är värre än många små fel
- Använd **MAE** när alla fel ska värderas lika mycket

### 3. Om vi ska rangordna olika modeller, spelar det någon roll om RMSE eller MSE används? Varför?

**Nej, det spelar INGEN roll för rangordningen!**

#### **Varför?**

**Matematisk förklaring:**

- RMSE = √MSE
- Kvadratroten är en **monoton funktion**
- Det betyder att om A > B, så är √A > √B

**Praktiskt exempel:**

```
Modell A: MSE = 100, RMSE = 10
Modell B: MSE = 144, RMSE = 12
Modell C: MSE = 81,  RMSE = 9
```

**Rangordning med MSE:** C (81) < A (100) < B (144)
**Rangordning med RMSE:** C (9) < A (10) < B (12)

**Resultat:** Samma rangordning!

#### **När väljer man vad?**

**Använd MSE när:**

- Du bara vill rangordna modeller
- Beräkningshastighet är viktigt
- Du optimerar hyperparametrar

**Använd RMSE när:**

- Du vill tolka felstorleken
- Du presenterar resultat för andra
- Du vill förstå praktisk betydelse

#### **Praktisk regel:**

1. **Under utveckling:** Använd MSE (snabbare)
2. **Vid presentation:** Använd RMSE (tolkbart)
3. **Rangordning:** Spelar ingen roll vilket du väljer

**Viktigt:** Detta gäller bara för MSE vs RMSE. MAE kan ge annan rangordning!

### 4. Förklara mycket översiktligt vad gradient descent är.

#### **Gradient Descent - Optimeringsalgoritm**

**Enkelt koncept:** Gradient descent är en algoritm som hittar **minimum** av en funktion genom att gå **nedåt längs lutningen**.

#### **Analogier för förståelse:**

**Kulle-analogin:**

- Tänk dig att du står på en kulle
- Du vill komma till botten (minimum)
- Du går i den riktning som är nedåt (negativ lutning)
- Du fortsätter tills du når botten

**Termostat-analogin:**

- Du vill hitta den perfekta temperaturen
- Om det är för varmt, sänker du temperaturen
- Om det är för kallt, höjer du temperaturen
- Du justerar tills det är perfekt

#### **I maskininlärning:**

**Syfte:** Hitta de bästa parametrarna för en modell

**Process:**

1. **Starta** med slumpmässiga parametrar
2. **Beräkna** hur fel modellen gör (loss function)
3. **Beräkna** lutningen (gradient) av felet
4. **Uppdatera** parametrarna i motsatt riktning av lutningen
5. **Upprepa** tills felet inte minskar mer

#### **Viktiga koncept:**

**Learning Rate (α):**

- Hur stora steg du tar
- För stort: kan hoppa över minimum
- För litet: tar lång tid att konvergera

**Konvergens:**

- När algoritmen hittar minimum
- Lutningen blir nära noll
- Modellen är tränad

#### **Varianter:**

1. **Batch Gradient Descent:**
   - Använder all data för varje steg
   - Stabil men långsam

2. **Stochastic Gradient Descent:**
   - Använder en observation i taget
   - Snabb men brusig

3. **Mini-batch Gradient Descent:**
   - Använder små grupper av data
   - Kompromiss mellan hastighet och stabilitet

#### **Praktisk betydelse:**

- **Automatisk träning:** Datorn hittar bästa parametrarna
- **Skalbarhet:** Fungerar för stora dataset
- **Flexibilitet:** Kan användas för många olika modeller

**Viktigt:** Du behöver inte implementera gradient descent själv - scikit-learn gör det åt dig!

### 5. Vad är the bias variance trade-off? Varför är mer komplexa modeller inte alltid bättre?

#### **Bias-Variance Trade-off**

**Grundkoncept:** Det finns en balans mellan modellens **bias** och **variance** som påverkar dess prestanda.

#### **Vad är Bias och Variance?**

**Bias (Fördom):**

- Fel från felaktiga antaganden om data
- Modellen är för enkel för att fånga sambanden
- **Underanpassning (Underfitting)**
- Exempel: Linjär modell för icke-linjära data

**Variance (Variation):**

- Fel från känslighet för träningsdata
- Modellen är för komplex och anpassar sig för mycket
- **Överanpassning (Overfitting)**
- Exempel: Polynom av hög grad som följer brus

#### **Trade-off-förhållandet:**

```
Enkel modell:     Hög Bias + Låg Variance
Komplex modell:   Låg Bias + Hög Variance
```

**Regel:** När du minskar bias, ökar variance och vice versa.

#### **Varför är komplexa modeller inte alltid bättre?**

**1. Överanpassning (Overfitting):**

- Komplexa modeller kan lära sig brus i träningsdata
- De generaliserar dåligt till ny data
- Högre variance leder till sämre prestanda på testdata

**2. Beräkningskostnad:**

- Komplexa modeller tar längre tid att träna
- Kräver mer minne och processorkraft
- Dyrare att använda i produktion

**3. Tolkningsbarhet:**

- Komplexa modeller är svåra att förstå
- Svårt att förklara varför de gör vissa prediktioner
- Mindre användbara för beslutstagning

**4. Instabilitet:**

- Små ändringar i data kan ge stora ändringar i modellen
- Mindre robusta och pålitliga

#### **Praktiskt exempel:**

**Linjär regression (enkel):**

- Hög bias: Kan inte fånga icke-linjära samband
- Låg variance: Stabil och robust

**Polynomregression (komplex):**

- Låg bias: Kan fånga komplexa samband
- Hög variance: Känslig för brus och outliers

#### **Hur hittar man rätt balans?**

**1. Korsvalidering:**

- Testa olika modellkomplexitet
- Hitta sweet spot mellan bias och variance

**2. Regularisering:**

- Ridge, Lasso, Elastic Net
- Minska variance genom att begränsa modellen

**3. Ensemble-metoder:**

- Random Forest, Bagging
- Kombinera flera modeller för bättre balans

#### **Praktisk regel:**

**"Börja enkelt, gör komplexare endast om nödvändigt"**

1. Börja med linjär regression
2. Om prestanda är dålig, prova mer komplexa modeller
3. Använd korsvalidering för att hitta optimal komplexitet
4. Kom ihåg: Enkel modell som fungerar är bättre än komplex modell som inte fungerar!

### 6. Några vanligt förekommande modeller för regressionsproblem är enligt nedan. Förklara översiktligt hur respektive modell fungerar.

#### **a) Linjär regression**

**Princip:** Modellerar ett linjärt samband mellan features och target.

**Formel:** y = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₚxₚ

**Fördelar:**

- Enkel och tolkbar
- Snabb att träna och använda
- Bra utgångspunkt

**Nackdelar:**

- Antar linjära samband
- Kan inte fånga komplexa mönster

**Användning:** Baseline-modell, när sambanden är linjära

#### **b) Ridge regression (L2-regularisering)**

**Princip:** Linjär regression med straff för stora koefficienter.

**Formel:** MSE + α × Σθᵢ²

**Effekt:**

- Krymper koefficienter mot 0
- Förhindrar överanpassning
- Minskar variance

**Användning:** När du har många features eller överanpassning

#### **c) Lasso regression (L1-regularisering)**

**Princip:** Linjär regression med straff för absoluta koefficienter.

**Formel:** MSE + α × Σ|θᵢ|

**Effekt:**

- Sätter vissa koefficienter till exakt 0
- Automatisk variabelselektion
- Sparar bara viktiga features

**Användning:** När du vill välja ut viktiga features

#### **d) Elastic Net**

**Princip:** Kombination av Ridge och Lasso.

**Formel:** MSE + r×α×Σ|θᵢ| + (1-r)/2×α×Σθᵢ²

**Fördelar:**

- Bästa av båda världarna
- Hanterar korrelerade features
- Flexibel regularisering

**Användning:** När du vill ha både variabelselektion och regularisering

#### **e) Support Vector Machines (SVM)**

**Princip:** Skapar en "väg" runt data med given bredd.

**Funktion:**

- Skapar en väg med bredd ε
- Innehåller så många observationer som möjligt
- Mitten-linjen blir prediktionen

**Fördelar:**

- Robust mot outliers
- Bra för små dataset
- Kan hantera icke-linjära samband (med kernels)

**Nackdelar:**

- Långsam för stora dataset
- Känslig för hyperparametrar

#### **f) Beslutsträd**

**Princip:** Delar data upp i regioner med ja/nej-frågor.

**Funktion:**

- Trädstruktur med noder och grenar
- Varje nod ställer en fråga om en feature
- Lövnoder ger prediktioner

**Fördelar:**

- Tolkbar och intuitiv
- Hanterar icke-linjära samband
- Kan hantera kategoriska features

**Nackdelar:**

- Instabil (små ändringar kan ge stora skillnader)
- Risk för överanpassning

#### **g) Ensemble learning**

**Princip:** Kombinerar flera modeller för bättre prestanda.

**Voting Regression:**

- Flera modeller röstar
- Tar medelvärde av prediktioner

**Bagging:**

- Skapar flera dataset med återläggning
- Tränar modeller på varje dataset
- Kombinerar prediktioner

**Fördelar:**

- Minskar variance
- Stabilare prediktioner
- Bättre generalisering

#### **h) Random Forest**

**Princip:** Ensemble av beslutsträd med bagging och slumpmässig variabelselektion.

**Funktion:**

- Skapar många beslutsträd
- Varje träd använder slumpmässig del av data och features
- Kombinerar prediktioner från alla träd

**Fördelar:**

- Mycket robust
- Bra prestanda
- Mindre risk för överanpassning
- Kan hantera saknade värden

**Nackdelar:**

- Svår att tolka
- Kan vara långsam
- Kräver mer minne

### 7. Vad menas med white box modeller och black box modeller?

#### **White Box Modeller**

**Definition:** Modeller där det är **enkelt att förstå och förklara** hur de kommer fram till sina prediktioner.

**Kännetecken:**

- **Tolkbara:** Du kan se exakt hur modellen fungerar
- **Transparenta:** Steg-för-steg förklaring möjlig
- **Intuitiva:** Lätt att förstå logiken

**Exempel:**

- **Linjär regression:** Du kan se koefficienterna och förstå vilken påverkan varje feature har
- **Beslutsträd:** Du kan följa vägen genom trädet och se varför en prediktion gjordes
- **Logistisk regression:** Du kan se odds-ratio och förstå sannolikheter

**Fördelar:**

- **Förtroende:** Användare förstår hur modellen fungerar
- **Felsökning:** Lätt att hitta problem
- **Regulering:** Uppfyller krav på förklarbarhet
- **Beslutstagning:** Stödjer mänskliga beslut

**Nackdelar:**

- Kan ha sämre prestanda än komplexa modeller
- Begränsad kapacitet att fånga komplexa samband

#### **Black Box Modeller**

**Definition:** Modeller där det är **svårt eller omöjligt att förstå** hur de kommer fram till sina prediktioner.

**Kännetecken:**

- **Ogenomträngliga:** Svårt att se inuti modellen
- **Komplexa:** Många lager och interaktioner
- **Abstrakta:** Logiken är inte direkt synlig

**Exempel:**

- **Neurala nätverk:** Många lager med komplexa vikter
- **Random Forest:** Många träd med komplexa interaktioner
- **SVM med kernels:** Transformationer som är svåra att visualisera
- **Ensemble-metoder:** Kombination av många modeller

**Fördelar:**

- **Hög prestanda:** Kan fånga komplexa samband
- **Flexibilitet:** Kan hantera många olika datatyper
- **Skalbarhet:** Fungerar bra med stora dataset

**Nackdelar:**

- **Lågt förtroende:** Användare förstår inte hur modellen fungerar
- **Svår felsökning:** Problem är svåra att identifiera
- **Regulatoriska problem:** Kan bryta mot krav på förklarbarhet
- **Etiska problem:** Kan göra diskriminerande beslut utan förklaring

#### **Praktiska överväganden:**

**När väljer man White Box?**

- Medicinska beslut
- Finansiella beslut
- Juridiska beslut
- När förklarbarhet är viktigare än prestanda

**När väljer man Black Box?**

- Rekommendationssystem
- Bildklassificering
- När prestanda är viktigare än förklarbarhet
- När modellen är komplext men fungerar bra

#### **Framtida trender:**

**Explainable AI (XAI):**

- Försök att göra black box-modeller mer förklarbara
- Tekniker som SHAP, LIME
- Balans mellan prestanda och förklarbarhet

**Hybrid-modeller:**

- Kombinera white box och black box
- Få bästa av båda världarna
- Ökande trend inom AI-forskning

### 8. Vad är skillnaden mellan bagging och pasting?

#### **Bagging (Bootstrap Aggregating)**

**Princip:** Skapar nya dataset genom **slumpmässigt urval med återläggning**.

**Process:**

1. Ta ett slumpmässigt urval från originaldata
2. **Samma observation kan väljas flera gånger** (återläggning)
3. Skapa ett nytt dataset med samma storlek som original
4. Träna en modell på det nya datasetet
5. Upprepa för flera modeller
6. Kombinera prediktioner (t.ex. medelvärde)

**Egenskaper:**

- **Med återläggning:** Samma observation kan förekomma flera gånger
- **Dataset-storlek:** Samma som original
- **Överlapp:** Många observationer förekommer i flera dataset
- **Variation:** Mindre variation mellan dataset

**Fördelar:**

- **Stabilare:** Mindre variation mellan modeller
- **Robust:** Fungerar bra med många modeller
- **Bevisad metod:** Väldokumenterad och testad

**Nackdelar:**

- **Mindre variation:** Dataset är mer lika varandra
- **Risk för överanpassning:** Om originaldata är liten

#### **Pasting**

**Princip:** Skapar nya dataset genom **slumpmässigt urval utan återläggning**.

**Process:**

1. Ta ett slumpmässigt urval från originaldata
2. **Varje observation kan bara väljas en gång** (utan återläggning)
3. Skapa ett nytt dataset (ofta mindre än original)
4. Träna en modell på det nya datasetet
5. Upprepa för flera modeller
6. Kombinera prediktioner

**Egenskaper:**

- **Utan återläggning:** Varje observation kan bara väljas en gång
- **Dataset-storlek:** Ofta mindre än original
- **Överlapp:** Mindre överlapp mellan dataset
- **Variation:** Större variation mellan dataset

**Fördelar:**

- **Större variation:** Dataset är mer olika varandra
- **Mindre överanpassning:** Särskilt bra för stora dataset
- **Snabbare träning:** Mindre dataset = snabbare träning

**Nackdelar:**

- **Mindre stabil:** Större variation mellan modeller
- **Kräver stora dataset:** Fungerar bäst med mycket data

#### **Jämförelse:**

| Egenskap | Bagging | Pasting |
|----------|---------|---------|
| **Urval** | Med återläggning | Utan återläggning |
| **Dataset-storlek** | Samma som original | Mindre än original |
| **Överlapp** | Hög | Låg |
| **Variation** | Låg | Hög |
| **Stabilitet** | Hög | Låg |
| **Prestanda** | Bra för många modeller | Bra för stora dataset |

#### **När väljer man vad?**

**Använd Bagging när:**

- Du har relativt lite data
- Du vill ha stabila resultat
- Du använder många modeller
- Du vill minimera variation

**Använd Pasting när:**

- Du har mycket data
- Du vill ha större variation mellan modeller
- Träningshastighet är viktigt
- Du vill undvika överanpassning

#### **Praktiskt exempel:**

**Bagging (Random Forest):**

- Varje träd tränas på ett dataset med återläggning
- Samma observationer kan förekomma i flera träd
- Ger stabila och robusta resultat

**Pasting:**

- Varje modell tränas på en unik del av data
- Inga observationer delas mellan modeller
- Ger större variation men kan vara mindre stabil

#### **Sammanfattning:**

**Bagging** är som att dra kort från en kortlek och lägga tillbaka dem - samma kort kan dras flera gånger.

**Pasting** är som att dela ut kort från en kortlek - varje kort kan bara delas ut en gång.

Båda metoderna har sina fördelar och används i olika situationer beroende på datamängd och mål.

## Resonemangsfrågor

### 9. Förklara hur man kan tolka figur 3.1 på sidan 113.

#### **Figur 3.1 - Enkel linjär regression**

**Vad visar figuren:**
Figur 3.1 visar en **schematisk bild över enkel linjär regression** med data, regressionslinje och residualer.

#### **Komponenter i figuren:**

**1. Svarta punkter (Data):**

- Representerar **observerade datapunkter** (yᵢ)
- Varje punkt är en observation med x-värde och y-värde
- Dessa är de verkliga värdena från datasetet

**2. Blå linje (Regressionslinje):**

- Representerar **predikterade värden** (ŷ = θ₀ + θ₁x)
- Visar vad modellen förutsäger för varje x-värde
- Den bästa linjära approximationen av data

**3. Röda vertikala linjer (Residualer):**

- Representerar **fel** mellan observerade och predikterade värden
- eᵢ = yᵢ - ŷᵢ (observerat - predikterat)
- Visar hur mycket modellen missar för varje datapunkt

#### **Tolkning av regressionslinjen:**

**Intercept (θ₀):**

- Var linjen skär y-axeln
- Predikterat värde när x = 0
- I praktiken: Baslinje-värde

**Lutning (θ₁):**

- Hur mycket y ökar när x ökar med 1 enhet
- Positiv lutning: Positivt samband
- Negativ lutning: Negativt samband
- Större lutning: Starkare samband

#### **Tolkning av residualerna:**

**Stora residualer:**

- Modellen missar mycket för dessa punkter
- Kan indikera outliers eller icke-linjära samband
- Punkter som är svåra att förutsäga

**Små residualer:**

- Modellen förutsäger väl för dessa punkter
- Punkter som följer det linjära sambandet
- Bra anpassning av modellen

**Mönster i residualerna:**

- Om residualerna följer ett mönster: Modellen kanske inte är lämplig
- Om residualerna är slumpmässiga: Bra anpassning

#### **Praktisk tolkning:**

**Exempel - Lön vs Ålder:**

- **Svarta punkter:** Verkliga löner för olika åldrar
- **Blå linje:** Predikterad lön baserat på ålder
- **Röda linjer:** Hur mycket modellen missar för varje person

**Insikter:**

- Positiv lutning: Lön ökar med ålder
- Stora residualer: Vissa personer tjänar mycket mer/mindre än förväntat
- Små residualer: Modellen förutsäger lön väl för de flesta

#### **Kvalitetsbedömning:**

**Bra anpassning:**

- Punkterna ligger nära linjen
- Små residualer
- Slumpmässiga residualer

**Dålig anpassning:**

- Punkterna spridda långt från linjen
- Stora residualer
- Mönster i residualerna

#### **Begränsningar:**

**Enkel linjär regression antar:**

- Linjärt samband mellan x och y
- Oberoende observationer
- Konstant varians (homoskedasticitet)
- Normalfördelade residualer

**Om dessa antaganden inte uppfylls:**

- Modellen kanske inte är lämplig
- Andra modeller kan vara bättre
- Transformationer kan behövas

#### **Sammanfattning:**

Figur 3.1 är en **grundläggande visualisering** som visar:

1. **Verkliga data** (svarta punkter)
2. **Modellens prediktioner** (blå linje)
3. **Modellens fel** (röda linjer)

Den hjälper oss att förstå hur väl modellen anpassar sig till data och var den missar.

### 10. Förklara hur man kan tolka figur 3.13 på sidan 140. Hur hänger den ihop med figur 3.14 på sidan 141?

#### **Figur 3.13 - Beslutsträd visualisering**

**Vad visar figuren:**
Figur 3.13 visar en **trädstruktur** med noder, grenar och beslut som beslutsträdet använder för att göra prediktioner.

#### **Komponenter i figur 3.13:**

**1. Rotnod (översta noden):**

- Startpunkt för prediktionen
- Innehåller första beslutet
- Exempel: "Är x₂ ≤ 0.438?"

**2. Interna noder:**

- Noder som ställer frågor
- Delar data upp i mindre grupper
- Exempel: "Är x₁ ≤ -0.654?"

**3. Lövnoder (ytersta noderna):**

- Slutpunkter för prediktionen
- Innehåller det slutliga predikterade värdet
- Exempel: "value = 1.837"

**4. Information i varje nod:**

- **Samples:** Antal observationer i noden
- **Squared error:** MSE för noden
- **Value:** Predikterat värde (endast i lövnoder)

#### **Hur prediktion fungerar:**

**Steg-för-steg process:**

1. **Starta i rotnoden**
2. **Ställ frågan** (t.ex. "Är x₂ ≤ 0.438?")
3. **Gå till vänster** om svaret är JA
4. **Gå till höger** om svaret är NEJ
5. **Upprepa** tills du når en lövnod
6. **Prediktera** värdet från lövnoden

**Exempel:**

- Observation: x₁ = -1, x₂ = 0.55
- Fråga 1: "Är x₂ ≤ 0.438?"  NEJ  Gå höger
- Fråga 2: "Är x₁ ≤ -0.654?"  JA  Gå vänster
- Resultat: Lövnod med value = 1.837

#### **Figur 3.14 - Beslutsträd i 2D-rum**

**Vad visar figuren:**
Figur 3.14 visar **samma beslutsträd** men visualiserat i det 2-dimensionella rummet med x₁ och x₂ som axlar.

#### **Komponenter i figur 3.14:**

**1. Datapunkter:**

- Svarta punkter som representerar observationer
- Varje punkt har koordinater (x₁, x₂)
- Färgkodning visar verkliga y-värden

**2. Beslutsgränser:**

- Vertikala och horisontella linjer
- Delar upp rummet i regioner
- Motsvarar besluten i figur 3.13

**3. Regioner:**

- Olika färgade områden
- Varje region motsvarar en lövnod
- Alla punkter i samma region får samma prediktion

**4. Färgskala:**

- Visar verkliga y-värden
- Hjälper att förstå datafördelningen
- Jämför med predikterade värden

#### **Samband mellan figurerna:**

**Korrespondens:**

- **Rotnod i 3.13**  **Första beslutsgränsen i 3.14**
- **Interna noder i 3.13**  **Ytterligare beslutsgränser i 3.14**
- **Lövnoder i 3.13**  **Regioner i 3.14**
- **Predikterade värden i 3.13**  **Färgade regioner i 3.14**

**Exempel-korrespondens:**

- Beslut "x₂ ≤ 0.438" i 3.13  Horisontell linje vid x₂ = 0.438 i 3.14
- Beslut "x₁ ≤ -0.654" i 3.13  Vertikal linje vid x₁ = -0.654 i 3.14
- Lövnod med value = 1.837 i 3.13  Region med motsvarande färg i 3.14

#### **Insikter från visualiseringen:**

**Trädstruktur (3.13):**

- Visar **logisk struktur** av besluten
- Lätt att följa prediktionsprocessen
- Visar hierarki av beslut

**2D-visualisering (3.14):**

- Visar **geometrisk struktur** av besluten
- Lätt att se hur data är uppdelad
- Visar komplexitet och anpassning

#### **Praktisk tolkning:**

**Fördelar med beslutsträd:**

- **Tolkbara:** Du kan se exakt varför en prediktion gjordes
- **Flexibla:** Kan hantera icke-linjära samband
- **Automatiska:** Väljer bästa beslut automatiskt

**Begränsningar:**

- **Stuckade beslut:** Kan bara dela upp rummet med vertikala/horizontella linjer
- **Instabila:** Små ändringar i data kan ge helt annorlunda träd
- **Överanpassning:** Kan bli för komplexa

#### **Sammanfattning:**

**Figur 3.13** visar beslutsträdet som en **logisk struktur** med noder och beslut.

**Figur 3.14** visar samma beslutsträd som en **geometrisk uppdelning** av datarummet.

Tillsammans ger de en komplett förståelse för hur beslutsträd fungerar både logiskt och geometriskt.
