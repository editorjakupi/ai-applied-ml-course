# Svar på Faktafrågor och Resonemangsfrågor - Kapitel 10: Chattbottar

## Faktafrågor

### 1. Vad är prompt engineering?

Prompt engineering innebär att vara så specifik, deskriptiv och detaljerad som möjligt om önskad kontext, utfall, längd, format, stil med mera när man ställer frågor till chattbottar. Ett gammalt talesätt säger "som man frågar får man svar", och detta gäller även chattbottar. Generellt sett kan vi ställa de frågor vi har och få rimliga svar utan att behöva tänka alltför mycket, men om vi vill ha bättre svar kan vi försöka ställa frågor på ett mer strukturerat sätt.

Exempel:
- Mindre effektiv: "skriv en dikt om AI"
- Mer effektiv: "Skriv en kort och inspirerande dikt om AI som fokuserar på dess möjligheter och risker inom utbildning. Stilen ska likna Shakespeares."

### 2. Vad är RAG?

RAG står för **Retrieval Augmented Generation** (hämtningsförstärkt generering). RAG är en teknik som låter oss anpassa chattbottens svar utifrån en given kontext, exempelvis utifrån egna dokument.

RAG-tekniken fungerar så att om vi ger modellen en kontext (ett eller flera dokument) och säger åt modellen att enbart svara utifrån den givna kontexten. Om modellen inte hittar informationen i kontexten ska den säga "Det vet jag inte" istället för att gissa.

En RAG-modell innehåller två delar:
1. **Retriever**: Söker efter relevanta stycken i en större text
2. **Generator**: Genererar svaren utifrån den givna kontexten

RAG-tekniken använder chunking (dela upp text i mindre delar), embeddings (numeriska representationer av texten), och semantisk sökning (ofta med cosinuslikhet) för att hitta de mest relevanta delarna av texten att använda som kontext.

### 3. Vad är chunking och embeddings?

**Chunking** är processen att dela upp en lång text i mindre delar, som brukar kallas chunks. Detta behövs eftersom vi småningom kommer att söka efter de delar av texten som innehåller den information vi är intresserade av, så kallad semantisk sökning. Det finns flera strategier för chunking:

1. **Fixed-length chunking**: Delar upp texten i ett antal lika stora delar baserat på tokens, ord eller tecken. Vanligtvis låter man delarna överlappa med ett antal tecken.

2. **Sentence-based chunking**: Delar upp texten i meningar med Pythons `split()`-funktion.

3. **Semantic chunking**: Börjar med sentence-based chunking, skapar embeddings av meningarna och jämför deras innehåll med en semantisk sökning. Utifrån resultaten skapas chunks som innehåller meningar som ligger nära varandra i betydelse.

**Embeddings** är numeriska representationer av texten. I RAG-sammanhang behöver vi skapa embeddings (numeriska representationer av texten) innan vi kan använda dem som kontext till vår RAG-modell. Embeddings fångar textens betydelse: chunks som ligger närmare varandra i betydelse också har embeddings som ligger närmare varandra rent matematiskt. Detta gör det möjligt att göra semantiska sökningar där vi hittar textstycken som är semantiskt lika vår fråga, även om de inte innehåller exakt samma ord.

## Resonemangsfrågor

### 4. Förklara översiktligt hur man kan evaluera en chattbot.

För att utvärdera om en chattbot (eller specifikt en RAG-modell) fungerar som vi vill kan vi använda följande metod:

1. **Skapa testdata**: Skriv ett antal frågor samt svar på frågorna (ideal answers eller ground truth).

2. **Låta modellen svara**: Låt modellen svara på frågorna.

3. **Utvärdera svaren**: Använd en annan modell (eller manuell bedömning) för att utvärdera om svaren är i linje med det vi önskar.

4. **Sätt betyg**: Sätt "betyg" på svaren, t.ex.:
   - 1 om väldigt nära det önskade svaret
   - 0.5 om delvis i linje
   - 0 om felaktigt

Detta ger oss data att jämföra olika modeller med och hjälper oss att förbättra chattboten systematiskt.

Andra aspekter att utvärdera kan vara:
- **Relevans**: Är svaren relevanta för frågan?
- **Korrekthet**: Är informationen korrekt?
- **Fullständighet**: Svarar modellen på hela frågan?
- **Klarhet**: Är svaren tydliga och begripliga?
- **Håller sig till kontexten**: För RAG-modeller, håller sig svaren till den givna kontexten?

### 5. Din kollega frågar dig vad ELIZA och Turingtestet är. Läs nedanstående två länkar innan du besvarar frågan.

**ELIZA** var en av de tidigaste chattbottarna, utvecklad av Joseph Weizenbaum vid MIT på 1960-talet. ELIZA simulerade en psykoterapeut och använde enkla regler för att omformulera användarens påståenden som frågor. Trots sin enkelhet kunde många människor bli övertygade om att de pratade med en verklig person. ELIZA visade tidigt på potentialen och begränsningarna hos konversationsbaserad AI.

**Turingtestet** är ett test som föreslogs av Alan Turing 1950 för att avgöra om en maskin kan tänkas. Testet går ut på att en människa (bedömaren) har konversationer med både en människa och en maskin via text, utan att veta vilken som är vilken. Om bedömaren inte kan skilja mellan människan och maskinen på ett tillförlitligt sätt, anses maskinen ha klarat testet. Turingtestet har varit en viktig referenspunkt i diskussionen om artificiell intelligens och vad det innebär att vara intelligent. Moderna chattbottar som ChatGPT har visat sig kunna lura många människor i liknande tester, vilket har återupplivat diskussionen om vad intelligens verkligen innebär.

### 6. Ge konkreta exempel på hur olika företag och organisationer kan ta nytta av chattbottar samt vilka risker som finns.

**Exempel på användningsområden:**

1. **Kundservice**: Företag kan använda chattbottar för att svara på vanliga frågor, hantera bokningar, och ge support dygnet runt. Detta kan minska kostnader och förbättra kundnöjdheten.

2. **Intern kunskapsbas**: Organisationer kan använda RAG-baserade chattbottar för att låta anställda söka i interna dokument, policyer och procedurer. Exempelvis kan en chattbot hjälpa anställda att hitta information om företagets HR-policyer.

3. **Utbildning**: Skolor och universitet kan använda chattbottar som läxhjälp, för att förklara koncept, eller för att ge feedback på studenters arbete.

4. **Hälsovård**: Chattbottar kan användas för att ge grundläggande hälsoråd, påminna patienter om medicinering, eller hjälpa till med att boka tider.

5. **E-handel**: E-handelsföretag kan använda chattbottar för att hjälpa kunder att hitta produkter, ge produktrekommendationer, och hantera returer.

**Risker:**

1. **Felaktig information**: Chattbottar kan ge felaktiga eller missvisande svar, särskilt om de inte håller sig till sin kontext (för RAG-modeller) eller om träningsdatan innehåller felaktigheter.

2. **Säkerhetsrisker**: Känslig information kan exponeras om chattbottar inte är korrekt konfigurerade, särskilt vid användning av molnbaserade modeller.

3. **Bias och diskriminering**: Chattbottar kan återspegla bias från träningsdatan, vilket kan leda till diskriminerande eller opassande svar.

4. **Beroende och dehumanisering**: Överdriven användning av chattbottar kan minska mänsklig interaktion och skapa beroende.

5. **Arbetsförlust**: Automatisering av kundservice och andra uppgifter kan leda till att människor förlorar sina jobb.

6. **Manipulation och desinformation**: Chattbottar kan användas för att sprida desinformation eller manipulera människor.

7. **Etiska problem**: Användning av chattbottar i känsliga sammanhang (som hälsovård eller juridik) kan leda till etiska problem om de inte används korrekt eller om ansvaret inte är tydligt definierat.

8. **Tekniska begränsningar**: Chattbottar förstår inte alltid kontexten korrekt och kan ge svar som verkar rimliga men är felaktiga eller opassande.

För att minska dessa risker är det viktigt att:
- Använda RAG för att begränsa svaren till specifik kontext
- Implementera korrekt säkerhet och dataskydd
- Ha mänsklig översyn och kvalitetskontroll
- Utbilda användare om chattbottens begränsningar
- Följa etiska riktlinjer och regelverk
