# Kapitel 6: Klustring - Faktafrågor

## Fråga 1: Vad är klustring för något? Ge några exempel på tillämpningsområden.

**Svar:**
Klustring är en oövervakad maskininlärningsteknik som grupperar liknande datapunkter tillsammans utan att veta de korrekta etiketterna i förväg. Målet är att hitta naturliga grupperingar i data.

**Exempel på tillämpningsområden:**

- **Kundsegmentering**: Gruppera kunder baserat på köpbeteende för anpassad marknadsföring
- **Bildkomprimering**: Reducera färgpalett med k-means för att behålla viktiga färger
- **Genanalys**: Gruppera gener med liknande uttryck för att identifiera genfunktioner
- **Dokumentklustring**: Organisera stora textsamlingar genom att gruppera liknande dokument
- **Medicinsk bildanalys**: Identifiera olika typer av celler eller vävnader
- **Säkerhetsanalys**: Detektera ovanliga mönster i nätverkstrafik
- **Rekommendationssystem**: Gruppera användare med liknande preferenser

## Fråga 2: Förklara översiktligt hur K-means fungerar. Använd figur 6.3 (sidan 238) och figur 6.4 (sidan 239) i din förklaring.

**Svar:**
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
