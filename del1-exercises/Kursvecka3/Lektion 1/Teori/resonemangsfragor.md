# Kapitel 4 - Klassificering: Resonemangsfrågor

## Fråga 1: Precision vs Recall Tradeoff

**Stina säger till Kalle på lunchsamtalet "jag vill ha högsta möjliga precision för vår klassificeringsmodell". Kalle funderar ett tag och säger "men vad händer då med recall"? Vad hade du svarat? I vilka fall kan man tänka sig vilja ha en så hög precision som möjligt? I vilka fall kan det vara dåligt? Om vi tänker oss rättsväsendet där en slutgiltig dom kan leda till fängelse, vad kan vi då säga om precision-recall tradeoff?**

### Svar:

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

---

## Fråga 2: Tolkning av Figur 4.8

**Förklara hur man kan tolka figur 4.8 på sidan 175.**

### Svar:

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

---

## Fråga 3: Fit vs Transform

**På sidan 209 står det "på träningsdatan använder vi `.fit_transform()`, på valideringsdatan och testdatan använder vi endast `.transform()`." Förklara logiken bakom detta.**

### Svar:

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
