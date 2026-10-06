# Svar på Faktafrågor och Resonemangsfrågor - Kapitel 8: CNN

## Faktafrågor

### 1. Vad krävs det för att ett neuralt nätverk ska klassas som CNN?

För att ett neuralt nätverk ska klassas som CNN (Convolutional Neural Network) behöver det minst ett convolution lager som ett av de dolda lagren. CNN är en klass av artificiella neurala nätverk (ANN), och det är närvaron av convolution layers som skiljer CNN från vanliga ANN-modeller.

### 2. Inom vilket tillämpningsområde är CNN generellt sett en väldigt kraftfull modell?

CNN är generellt sett väldigt kraftfulla inom datorseende (computer vision). CNN inspirerades av den del av hjärnan som hanterar vision eller datorseende. Som tumregel kan vi säga att om vi arbetar med visuell data (bilder eller videor) så är CNN den modell vi börjar med. CNN presterar mycket bra när vi arbetar med bilder eller videor, exempelvis för klassificering där man vill identifiera vad det är på en bild.

### 3. Vad menas med RGB?

RGB står för Röd, Grön och Blå. Det är tre kanaler som representerar grundfärgerna i additiv färgblandning. När vi arbetar med färgbilder i CNN så består varje bild ofta av tre kanaler som representerar dessa färger. Exempelvis kan en bild ha storleken (32 x 32) och tre kanaler (RGB), vilket ger oss en bild med dimensionerna (32, 32, 3).

### 4. Vad är data augmentation?

Data augmentation är en användbar teknik när vi arbetar med bild-data. Det innebär att vi förvränger bilder på olika sätt, exempelvis:
- Roterar dem
- Ändrar ljusstyrkan
- Zoomar in/ut

Syftet med data augmentation är att få fler bilder till vår träningsdata, vilket skyddar modellen mot att bli överanpassad. Detta gör det till en form av regularisering. Viktigt är att bilderna endast förvrängas så pass lite att en människas förmåga att tolka bilden inte påverkas.

## Resonemangsfrågor

### 5. Förklara översiktligt hur CNN fungerar. Använd begreppen convolution layer och pooling layer i ditt svar.

CNN fungerar genom att först identifiera low-level features som enklare färger och former. Dessa mer enkla egenskaper kombineras därefter för att identifiera high-level features såsom ögon, mun, öron och dylikt.

Modellen använder convolutional layers, ofta i kombination med pooling layers.

**Convolution layers:**
Ett convolutional lager består av flera filter där varje filter "söker efter" och betonar vissa lokala attribut/egenskaper. I praktiken används flera filter för att hitta flera olika attribut, och när nätverket tränas så lär det sig vilka filter som är bra att använda. Filtret tillämpas på varje delmatris i bilden genom elementvis multiplikation och addition. Om en delmatris i originalbilden liknar filtret så kommer vi få ett högt värde; annars ett litet värde. Således fokuserar convolved image på de delar i en bild som liknar filtret som tillämpas.

**Pooling layers:**
Pooling layers fokuserar på de viktigaste delarna i en bild. Resultatet blir att bildens storlek minskar. Max pooling layer tar maximumvärdet för varje icke-överlappande delmatris. Om vi använder ett (2 x 2) max pooling tas alltså maximumvärdet för varje icke-överlappande (2 x 2) delmatris.

Genom att kombinera convolution layers och pooling layers kan CNN gradvis bygga upp mer komplexa representationer av bilden, från enkla mönster till komplexa objekt.

### 6. Din kollega ber dig förklara figur 8.4. Gör det!

Figur 8.4 visar en exempel på en CNN-modellarkitektur för 100 klasser. Här är en steg-för-steg förklaring:

1. **Input-lagret**: Vi ser en bild med storleken (32 x 32) som består av tre kanaler som representerar färgerna röd, grön och blå (RGB). Generellt sett används dessa som grundfärger i additiv färgblandning.

2. **Convolution layer**: För varje kanal tillämpas ett convolution lager med två filter. Därför får vi nu sex kanaler totalt (3 kanaler × 2 filter = 6 kanaler).

3. **Max pooling**: Max pooling tillämpas där storleken (2 x 2) används för filtreringen. Den nya bildstorleken blir därför (16 x 16).

4. **Convolution layer**: Ett convolution lager med två filter tillämpas återigen vilket ger oss 12 stycken kanaler.

5. **Max pooling**: Max pooling tillämpas där storleken (2 x 2) används för filtreringen. Den nya bildstorleken blir därför (8 x 8).

6. **Convolution layer**: Ett convolution lager med två filter tillämpas återigen vilket ger oss 24 stycken kanaler.

7. **Max pooling**: Max pooling tillämpas där storleken (2 x 2) används för filtreringen. Den nya bildstorleken blir därför (4 x 4) vilket gör att varje bild består av totalt 16 pixlar.

8. **Flatten layer**: Ett flatten layer tillämpas så att pixlarna från de två dimensionella bilderna tas ut till en dimension.

9. **Output-lager**: I sista steget har vi ett output-lager med 100 stycken noder eftersom vi antog att datan vi arbetar med består av 100 klasser. Vi använder softmax som aktiveringsfunktion när vi arbetar med multiklass klassificering.

Notera att vi använder fler filter för varje efterföljande Conv2D-lager (2, 2, 2 filter i exemplet, men i praktiken ofta 32, 64, 128, 256). Anledningen är att efter varje MaxPooling2D-lager så sjunker dimensionen på bilderna, vilket kompenseras med fler kanaler.
