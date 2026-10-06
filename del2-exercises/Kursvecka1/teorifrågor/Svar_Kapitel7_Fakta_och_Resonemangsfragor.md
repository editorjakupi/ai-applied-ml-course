# Svar på Faktafrågor och Resonemangsfrågor - Kapitel 7: Artificiella Neurala Nätverk (ANN)

## FAKTAFRÅGOR

### 1. Vad har ANN modellerna inspirerats av?

ANN-modellerna (Artificiella Neurala Nätverk) har inspirerats av hur den mänskliga hjärnan fungerar med dess neuronnät. I presentationstalet för Nobelpriset i fysik 2024 sägs det att årets Nobelpristagare, John Hopfield och Geoffrey Hinton, inspirerades av neuronnätverket i den mänskliga hjärnan när de utvecklade artificiella neurala nätverk.

**Liknelsen mellan biologiska neuroner och artificiella neurala nätverk:**

- **Input layer**: Motsvarar sinnesintryck - vi får in något (t.ex. se en farlig tiger)
- **Hidden layer**: Bearbetning av det som kommit in - "dolda" tankeprocesser (t.ex. tänka att tigern är farlig)
- **Output layer**: Slutsats efter bearbetning - vi får ut något (t.ex. beslutet att springa)

Det är viktigt att notera att liknelsen med hjärnan är användbar för intuition, men ANN fungerar faktiskt inte på samma sätt som hjärnan - det är en inspiration, inte en exakt kopia.

---

### 2. Vad refererar "djup" till i begreppet "djupinlärning"?

"Djup" i begreppet "djupinlärning" (Deep Learning, DL) refererar till det faktum att neurala nätverk kan ha många dolda lager (hidden layers). Det finns ingen exakt definition på hur många lager som krävs för att ett neuralt nätverk ska klassas som "djupt", men i praktiken anses ett neuralt nätverk med flera dolda lager vara "djupt".

**Viktiga punkter:**
- Djupinlärning = neurala nätverk med många dolda lager
- Begreppen "djupinlärning" och "neurala nätverk" används ofta synonymt
- Ett single-layer perceptron har inga dolda lager
- Ett Multilayer Perceptron (MLP) har flera dolda lager och kan därför klassas som djupinlärning

---

### 3. Förklara vad som händer i figur 7.2, 7.3, figur 7.4 och figur 7.5. Varför används aktiveringsfunktioner?

#### Figur 7.2: Enkel linjär regression
- Visar ett neuralt nätverk utan aktiveringsfunktioner
- Input-noderna multipliceras med vikter (W1, W2) och adderar en bias
- Outputen blir: y = W1 + W2·x1
- **Problemet**: Detta är exakt samma som linjär regression - vi har inte fått något nytt!

#### Figur 7.3: Mer komplex arkitektur (fortfarande utan aktiveringsfunktioner)
- Har ett dolt lager med flera noder
- Trots den ökade komplexiteten blir outputen fortfarande: y = θ1 + θ2·x1
- **Problemet**: Även med dolda lager får vi samma resultat som linjär regression om vi inte använder aktiveringsfunktioner

#### Figur 7.4: Aktiveringsfunktioner
- Visar två viktiga aktiveringsfunktioner:
  - **Logistic/Sigmoid**: σ(z) = 1 / (1 + e^(-z)) - ger värden mellan 0 och 1
  - **ReLU (Rectified Linear Unit)**: ReLU(z) = max(0, z) - ger värden ≥ 0
- Visar hur dessa funktioner ser ut grafiskt

#### Figur 7.5: Single-layer perceptron med aktiveringsfunktioner
- Visar en arkitektur som faktiskt skiljer sig från linjär regression
- Karakteriseras av:
  - Feedforward (data går framåt från input till output)
  - Fully connected/dense (alla noder är kopplade)
  - **Icke-linjära aktiveringsfunktioner** (detta är nyckeln!)

**Varför används aktiveringsfunktioner?**

Aktiveringsfunktioner introducerar **icke-linearitet** i neurala nätverk. Utan aktiveringsfunktioner skulle ett neuralt nätverk bara kunna modellera linjära samband - precis som linjär regression. Med icke-linjära aktiveringsfunktioner kan nätverket lära sig komplexa, icke-linjära samband mellan input och output.

**Två viktiga aktiveringsfunktioner:**
1. **Logistic/Sigmoid**: Används för binär klassificering (kan tolkas som sannolikhet)
2. **ReLU**: Standardval för hidden layers i de flesta fall

---

### 4. Har neurala nätverk få eller många parametrar?

Neurala nätverk har **väldigt många parametrar**. Detta är en av de mest karakteristiska egenskaperna hos neurala nätverk.

**Exempel:**
I ett exempel med MLPRegressor med:
- 20 input features
- Hidden layers: (200, 100, 10)
- 1 output

Får vi totalt **25,321 parametrar**:
- 25,010 vikter (weights)
- 311 bias-värden

**Jämförelse:**
- En linjär regressionsmodell med 20 features har bara 21 parametrar (20 vikter + 1 bias)
- Neurala nätverk har ofta tusentals eller miljoner parametrar

**Konsekvenser av många parametrar:**
1. **Mer data behövs**: Ju fler parametrar, desto mer data behövs för att träna modellen bra
2. **Ökad risk för överanpassning**: Många parametrar ökar risken att modellen blir överanpassad (overfitted) eftersom modellen blir för komplex i förhållande till datans faktiska komplexitet
3. **Behov av regularisering**: Därför behövs tekniker som dropout, L1/L2 regularisering, early stopping och batch normalization

---

### 5. Förklara intuitivt hur dropout-regularisering fungerar.

**Dropout** är en populär metod för att regularisera neurala nätverk och minska överanpassning.

**Hur det fungerar:**
- Vid varje träningsiteration har varje neuron en sannolikhet **p** (vanligtvis 10-50%) att bli "droppad"
- En droppad neuron har output 0 och deltar inte i beräkningarna under den iterationen
- Detta sker slumpmässigt för varje iteration

**Intuition:**
Detta tvingar neuronerna att "lära sig själva" och inte förlita sig för mycket på andra neuroner. Det förhindrar så kallad **co-adaptation** där neuroner blir beroende av varandra på ett sätt som kan leda till överanpassning.

**Praktiskt:**
- Om modellen är överanpassad: höj dropout-graden (t.ex. från 0.3 till 0.5)
- Vanliga värden: 10-50% (0.1 till 0.5)
- Dropout används endast under träning, inte vid prediktion

**Exempel:**
Om vi har dropout rate = 0.3 (30%), så kommer vid varje träningsiteration 30% av neuronerna slumpmässigt att stängas av. Detta gör att nätverket måste lära sig att fungera även när vissa neuroner saknas, vilket gör modellen mer robust.

---

## RESONEMANGSFRÅGOR

### 6. Din kollega ber dig förklara tabell 7.1 och tabell 7.2. Gör det!

#### Tabell 7.1: Typisk MLP-arkitektur för regressionsproblem

**Antal input neuroner:**
- En per input feature
- Exempel: För MNIST (28x28 pixlar) = 784 input neuroner

**Antal hidden layers:**
- Beror på problemet, men vanligtvis 1 till 5 lager

**Antal neuroner per hidden layer:**
- Beror på problemet, men vanligtvis 10 till 100

**Antal output neuroner:**
- 1 (eller flera om flera dimensioner ska predikteras)

**Hidden activation:**
- ReLU (eller SELU) - standardval för dolda lager

**Output activation:**
- Ingen aktiveringsfunktion, eller
- ReLU (när y är positivt - t.ex. ålder, lön), eller
- Logistisk/tanh (när y är i ett begränsat intervall)

**Loss-funktion:**
- MSE (Mean Squared Error) eller
- MAE (Mean Absolute Error) om det finns outliers

**Exempel på användning:**
Om vi vill prediktera någons ålder (som alltid är ≥ 0), kan vi använda ReLU som output activation eftersom ReLU alltid ger värden ≥ 0.

---

#### Tabell 7.2: Typisk MLP-arkitektur för klassificeringsproblem

Tabellen visar tre typer av klassificeringsproblem:

**A. Binär klassificering:**
- **Output neuroner**: 1
- **Output activation**: Logistic (sigmoid)
- **Loss-funktion**: Cross entropy
- **Exempel**: Har personen en hund? (Ja/Nej)

**B. Multilabel binär klassificering:**
- **Output neuroner**: 1 per label
- **Output activation**: Logistic (sigmoid) för varje output
- **Loss-funktion**: Cross entropy
- **Exempel**: Vilka djur har personen? (Kan ha både hund OCH katt)

**C. Multiklass klassificering:**
- **Output neuroner**: 1 per klass
- **Output activation**: Softmax (generalisering av sigmoid där summan av alla outputs blir 1)
- **Loss-funktion**: Cross entropy
- **Exempel**: Vilken siffra är det? (0, 1, 2, ..., 9 - endast EN siffra)

**Gemensamt för alla klassificeringsproblem:**
- Antal input neuroner: Samma som för regression (en per feature)
- Antal hidden layers och neuroner: Samma som för regression (1-5 lager, 10-100 neuroner)

**Viktig skillnad:**
- **Softmax** används för multiklass eftersom den ger en sannolikhetsfördelning där summan av alla outputs = 1.0
- **Logistic (sigmoid)** används för binär klassificering eftersom den ger ett värde mellan 0 och 1 som kan tolkas som sannolikhet

---

### 7. Experimentera med neurala nätverk på följande länk: 
[https://playground.tensorflow.org/](https://playground.tensorflow.org/)

**Instruktioner:**
Denna uppgift kräver att du själv går in på TensorFlow Playground och experimenterar. Här är några saker du kan prova:

**Experiment att göra:**
1. **Testa olika aktiveringsfunktioner**: Jämför ReLU, Tanh, Sigmoid - se hur de påverkar modellens förmåga att lära sig
2. **Variera antal hidden layers**: Testa 1, 2, 3, 4 lager - se hur djupare nätverk kan lära sig mer komplexa mönster
3. **Variera antal neuroner per lager**: Se hur fler neuroner kan hjälpa modellen
4. **Testa olika datasets**: Prova de olika datamönstrena (spiral, cirkel, XOR, etc.)
5. **Lägg till noise**: Se hur modellen hanterar brusig data
6. **Observera learning rate**: Se hur learning rate påverkar träningsprocessen

**Vad du ska lära dig:**
- Intuition för hur neurala nätverk fungerar visuellt
- Hur olika hyperparametrar påverkar träningen
- Hur modellen lär sig att separera olika klasser
- Hur aktiveringsfunktioner introducerar icke-linearitet

**Reflektion:**
Efter att ha experimenterat, reflektera över:
- När behövs fler lager?
- När behövs fler neuroner?
- Hur påverkar learning rate träningen?
- Vilka mönster är svårast att lära sig?

---

### 8. Förklara översiktligt hur backpropagation fungerar. Använd figur 7.7 till din hjälp.

**Backpropagation** är algoritmen som används för att träna neurala nätverk. Den är grunden för hur neurala nätverk lär sig.

**Steg-för-steg förklaring:**

#### 1. Batch-hantering
- Algoritmen hanterar en batch av data (t.ex. 32 observationer) åt gången
- Detta gör träningen mer effektiv än att hantera en observation i taget

#### 2. Forward pass (framåtpassering)
- Input-datan färdas framåt genom nätverket
- Varje lager beräknar sina värden baserat på föregående lager
- Man gör en prediktion baserat på nuvarande vikter

#### 3. Loss-beräkning
- Felet mäts genom att jämföra sanna värden med predikterade värden
- Detta görs via loss-funktionen (t.ex. MSE för regression, Cross entropy för klassificering)
- Loss-funktionen ger ett mått på hur fel modellen var

#### 4. Backward pass (bakåtpassering) - "backpropagation"
- Algoritmen går **tillbaka** genom nätverket (därav namnet "backpropagation")
- Den beräknar hur varje vikt har bidragit till det totala felet
- Detta görs genom att använda kedjeregeln (chain rule) från kalkylen
- Varje vikt får ett "ansvar" för felet - vikter som bidrog mycket till felet får större ansvar

#### 5. Gradient Descent
- Baserat på "ansvaret" för felet, justeras vikterna så att felet minskar
- Learning rate bestämmer hur stora justeringar som görs
- Vikterna uppdateras i riktning som minskar felet

**Intuition:**
Tänk dig att du skjuter på en måltavla:
- **Forward pass**: Du skjuter och ser var pilen landade
- **Loss-beräkning**: Du mäter hur långt från mittpunkten pilen landade
- **Backpropagation**: Du analyserar vilka faktorer (vikter) som gjorde att pilen hamnade fel
- **Gradient Descent**: Du justerar ditt siktande (vikterna) för att nästa skott ska bli bättre

**Varför "backward"?**
Algoritmen måste gå bakåt eftersom felet uppstår i output-lagret, men vi behöver veta hur alla lager bidrog till felet. Genom att gå bakåt kan vi sprida "ansvaret" för felet tillbaka genom alla lager.

**Viktiga begrepp:**
- **Gradient**: Riktingen och storleken på förändringen som behövs för att minska felet
- **Chain rule**: Matematisk regel som gör det möjligt att beräkna gradienter i komplexa nätverk
- **Learning rate**: Hur stora steg vi tar när vi uppdaterar vikterna

**Praktiskt:**
- Backpropagation är automatiserad i Keras/TensorFlow - vi behöver inte implementera den själva
- Men förståelsen för hur den fungerar hjälper oss att förstå varför vissa saker fungerar (t.ex. varför vi behöver icke-linjära aktiveringsfunktioner)

---

## Sammanfattning

**Faktafrågor** handlar om grundläggande kunskaper om ANN:
- Inspiration från hjärnan
- Vad "djup" betyder
- Varför aktiveringsfunktioner behövs
- Att neurala nätverk har många parametrar
- Hur dropout fungerar

**Resonemangsfrågor** kräver djupare förståelse:
- Tolka och förklara riktlinjer för modellarkitektur (tabellerna)
- Experimentera och lära sig praktiskt
- Förstå grundläggande algoritmer (backpropagation)

Genom att besvara dessa frågor får man en solid grund för att arbeta med neurala nätverk i praktiken.
