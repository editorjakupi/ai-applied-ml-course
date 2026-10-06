# Svar på Faktafrågor och Resonemangsfrågor - Kapitel 6: Klustring

## Faktafrågor

### 1. Vad är klustring för något? Ge några exempel på tillämpningsområden.

Klustring är en oövervakad maskininlärningsteknik som grupperar liknande datapunkter tillsammans utan att veta de korrekta etiketterna i förväg. Målet är att hitta naturliga grupperingar i data.

**Exempel på tillämpningsområden:**

- **Kundsegmentering**: Gruppera kunder baserat på köpbeteende för anpassad marknadsföring
- **Bildkomprimering**: Reducera färgpalett med k-means för att behålla viktiga färger
- **Genanalys**: Gruppera gener med liknande uttryck för att identifiera genfunktioner
- **Dokumentklustring**: Organisera stora textsamlingar genom att gruppera liknande dokument
- **Medicinsk bildanalys**: Identifiera olika typer av celler eller vävnader
- **Säkerhetsanalys**: Detektera ovanliga mönster i nätverkstrafik
- **Rekommendationssystem**: Gruppera användare med liknande preferenser

### 2. Förklara översiktligt hur K-means fungerar. Använd figur 6.3 (sidan 238) och figur 6.4 (sidan 239) i din förklaring.

K-means är en iterativ klusteralgoritm som fungerar enligt följande steg:

**Algoritm:**

1. **Initialisering**: Välj k centroider slumpmässigt (figur 6.3 visar initiala centroider)
2. **Tilldelning**: Tilldela varje datapunkt till närmaste centroid
3. **Uppdatering**: Beräkna nya centroider som medelvärde av tilldelade punkter
4. **Iteration**: Upprepa steg 2-3 tills konvergens (figur 6.4 visar slutresultat)

**Matematisk formulering:**

```
J = Σ(i=1 to k) Σ(x∈Ci) ||x - μi||²
```

där J är Within-cluster sum of squares (WCSS), k är antal kluster, Ci är kluster i, och μi är centroid för kluster i.

**Figurerna visar:**

- Figur 6.3: Initiala centroider placerade slumpmässigt
- Figur 6.4: Slutliga centroider efter konvergens med optimerade klustertilldelningar

**Fördelar:**

- Enkel och snabb algoritm
- Skalar bra med stora dataset
- Fungerar bra med sfäriska kluster

**Nackdelar:**

- Kräver att antal kluster (k) specificeras i förväg
- Känslig för initialisering
- Fungerar bara med sfäriska kluster
- Känslig för outliers

## Resonemangsfrågor

### 3. Hur kan man välja vilket antal kluster som ska användas för en K-means modell? Använd inertia, silhouette score och silhouette diagram i ditt svar.

Val av antal kluster är en kritisk aspekt av K-means klustring. Här är de huvudsakliga metoderna:

#### 1. Inertia (Elbow-metoden)

- **Inertia** = Within-cluster sum of squares (WCSS)
- Plotta inertia mot antal kluster
- Välj punkt där kurvan "böjer" (elbow point)
- **Fördelar**: Enkel att implementera
- **Nackdelar**: Subjektiv, inte alltid tydlig "elbow"

#### 2. Silhouette Score

- Mäter kvalitet på klustertilldelning
- Värden mellan -1 och 1
- Högre värden = bättre klustring
- **Beräkning för varje punkt**:
  - a(i) = genomsnittligt avstånd till andra punkter i samma kluster
  - b(i) = genomsnittligt avstånd till närmaste kluster
  - s(i) = (b(i) - a(i)) / max(a(i), b(i))
- **Fördelar**: Objektiv mått, tar hänsyn till både kompakthet och separation
- **Nackdelar**: Beräkningskrävande

#### 3. Silhouette Diagram

- Visar silhouette score för varje datapunkt
- Längd på staplar = silhouette score
- Färg = klustertillhörighet
- **Tolkning**:
  - Långa staplar = bra klustertilldelning
  - Korta staplar = osäker klustertilldelning
  - Negativa värden = fel tilldelning

#### Praktisk rekommendation:

1. Börja med Elbow-metoden för grov uppskattning
2. Använd silhouette score för finjustering
3. Visualisera med silhouette diagram för detaljerad analys
4. Överväg domänkunskap och affärsmål

### 4. Om du kollar på figur 6.10 på sidan 247, hur många kluster hade du valt och varför? Är det en "exakt vetenskap" att välja antalet kluster?

Baserat på figur 6.10 (som visar silhouette score mot antal kluster) skulle jag välja **k=3** eftersom:

#### Anledningar till k=3:

1. **Högst silhouette score**: K=3 ger det högsta silhouette score-värdet
2. **Tydlig topp**: Det finns en tydlig topp vid k=3, vilket indikerar optimal klusterkvalitet
3. **Balans**: K=3 ger en bra balans mellan kompakthet och separation

#### Är det en "exakt vetenskap"?

**Nej, det är inte en exakt vetenskap** av följande anledningar:

1. **Subjektivitet**: Olika mått kan ge olika resultat
2. **Domänkunskap**: Affärsmål och domänkunskap påverkar valet
3. **Dataegenskaper**: Olika dataset kräver olika antal kluster
4. **Kompromisser**: Måste balansera modellkomplexitet mot tolkbarhet
5. **Iterativ process**: Ofta behöver man testa flera alternativ

#### Praktiska överväganden:

- **Tolkbarhet**: Färre kluster är ofta lättare att tolka
- **Affärsmål**: Antal kluster ska matcha affärsbehov
- **Dataqualitet**: Bättre data = mer tillförlitliga resultat
- **Validering**: Testa resultatet på nya data

### 5. Hur tolkar man figur 6.13 på sidan 251?

Figur 6.13 visar ett **silhouette diagram** som hjälper till att bedöma klusterkvalitet. Här är hur man tolkar det:

#### Struktur av diagrammet:

- **Y-axeln**: Datapunkter (sorterade per kluster)
- **X-axeln**: Silhouette score (-1 till 1)
- **Färger**: Olika kluster
- **Staplar**: Silhouette score för varje datapunkt

#### Tolkning av resultat:

##### 1. **Längd på staplar**:

- **Långa staplar**: Bra klustertilldelning
- **Korta staplar**: Osäker klustertilldelning
- **Negativa staplar**: Fel tilldelning (punkt borde vara i annat kluster)

##### 2. **Färgkonsistens**:

- **Enhetlig färg per kluster**: Bra separation
- **Blandade färger**: Osäker klustertilldelning

##### 3. **Klusterstorlek**:

- **Jämna storlekar**: Balanserade kluster
- **Ojämna storlekar**: Kan indikera problem

##### 4. **Genomsnittlig silhouette score**:

- **Högt värde (>0.5)**: Bra klustring
- **Medelvärde (0.2-0.5)**: Acceptabel klustring
- **Lågt värde (<0.2)**: Dålig klustring

#### Praktiska insikter:

- **Kluster med många negativa värden**: Överväg att minska antal kluster
- **Kluster med varierande längder**: Kan indikera att klustret inte är homogent
- **Jämna staplar**: Indikerar stabil klustring

#### Förbättringsåtgärder:

1. **Justera antal kluster**: Testa fler/färre kluster
2. **Förbättra data**: Standardisera eller rensa data
3. **Välj annan algoritm**: Testa hierarkisk klustring eller DBSCAN
4. **Feature engineering**: Skapa bättre features
