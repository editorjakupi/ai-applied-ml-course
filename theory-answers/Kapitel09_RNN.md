# Svar på Faktafrågor och Resonemangsfrågor - Kapitel 9: RNN

## Faktafrågor

### 1. Rent generellt, vad är RNN-modeller designade för att användas till?

RNN-modeller är designade för att hantera sekventiell data såsom:
- Text
- Tal
- Tidsserier

I alla dessa fall är ordningen av datan viktig. Exempelvis har texten "Jag gillar att lära mig nya saker" en helt annan innebörd än "mig saker jag nya att gillar lära". RNN-modeller har ett minne som gör att sekventiell data där ordningen spelar roll kan hanteras.

### 2. Vad heter två populära modellarkitekturer inom RNN?

Två populära modellarkitekturer inom RNN är:
- **LSTM** (Long Short-Term Memory)
- **GRU** (Gated Recurrent Unit)

LSTM är generellt sett den mest använda RNN-arkitekturen. LSTM-arkitekturer har ett mer avancerat minne och kan lära sig vad som är viktigt att komma ihåg och vad som kan glömmas bort. GRU är en förenklad variant av LSTM men har i många sammanhang visat sig fungera lika bra som LSTM.

### 3. Vad står NLP för? Vad är embeddings?

**NLP** står för **Natural Language Processing** (naturlig språkbehandling på svenska). NLP handlar om att hantera mänskligt språk med hjälp av datorer. Det finns flera tillämpningsområden, exempelvis sentimentanalys där man klassificerar om en film anses vara bra eller dålig beroende på dess omdöme.

**Embeddings** (ord-embeddings eller word embeddings) är numeriska representationer av ord. För att kunna använda text-data i våra modeller måste vi först transformera texten till siffror. Istället för one-hot-encoding (som ger en hög-dimensionell matris som består av mestadels nollor) använder vi word-embedding. Det innebär att ord representeras som numeriska vektorer där vektorerna ska reflektera ordens semantiska betydelse. Ord som ligger nära varandra i betydelse ska också ha embeddings som ligger närmare varandra rent matematiskt. Exempelvis är "Varg" och "Hund" lika eftersom båda är djur, medan "Tiger" och "Katt" också är lika eftersom båda är vilddjur. Avståndet mellan dessa ord i embedding-rummet reflekterar deras semantiska likhet.

I praktiken kan vi använda ett embedding layer där vikterna tränas från datan, eller använda förtränade embeddings såsom Word2vec eller GloVe.

## Resonemangsfrågor

### 4. Din kollega ber dig förklara figur 9.2. Gör det!

Figur 9.2 visar en visualisering av hur en RNN-modell ser ut genom tid (på engelska brukar detta beskrivas som "unfolded in time").

I figuren ser vi hur outputen från tidigare tidssteg används som input till nästa tidssteg. Specifikt:
- Outputen y(t-3) är resultatet av inputen x(t-3)
- Outputen y(t-2) är resultatet av inputen x(t-2) och outputen y(t-3)
- Outputen y(t-1) är resultatet av inputen x(t-1) och outputen y(t-2)
- Outputen y(t) är resultatet av inputen x(t) och outputen y(t-1)

Detta visar att outputen för ett tidssteg beror på outputen i föregående tidssteg (som i sin tur beror på outputen i föregående tidssteg). Detta skapar ett minne i RNN-modellen, vilket möjliggör att sekventiell data där ordningen spelar roll kan hanteras.

Detta är skillnaden mellan RNN och feedforward neurala nätverk (ANN och CNN). I feedforward-nätverk flödar datan bara framåt från input till output, medan RNN har återkoppling där output från tidigare tidssteg används som input till nästa tidssteg.
