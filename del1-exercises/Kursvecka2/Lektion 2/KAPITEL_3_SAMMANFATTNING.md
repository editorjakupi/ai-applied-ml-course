#  KAPITEL 3 - REGRESSION

## En omfattande pedagogisk sammanfattning

---

##  VAD ÄR REGRESSION?

**Regression** är när vi försöker förutsäga ett **kontinuerligt värde** (som en siffra) baserat på andra variabler.

###  Vardagliga exempel från boken:

- **Bilvärdering**: Prediktera försäljningspriset på en bil beroende på miltal, ålder, bränsletyp och märke
- **Fastighetsvärdering**: Prediktera försäljningspriset på en fastighet beroende på storlek, byggnadsår, geografiskt läge och antal rum
- **Jordbruksproduktion**: Prediktera hur stor skörden (exempelvis potatis) blir i kilogram baserat på temperatur, mängden regn och bruket av gödningsmedel
- **Försäkringar**: Prediktera förväntad skadekostnad i kronor för en kund baserat på ålder, utbildning, geografiskt läge och tidigare historik
- **Efterfrågan på produkter**: Prediktera hur många enheter av en produkt som kommer säljas baserat på pris, säsong, kampanjer och tidigare försäljningsdata
- **Lönenivå**: Prediktera en persons lön i kronor beroende på utbildningsnivå, ålder, erfarenhet, bransch och geografiskt område

###  **Matematisk representation:**

- **Beroende variabel**: 𝑦 (vad vi vill prediktera)
- **Oberoende variabler**: 𝑥₁, 𝑥₂, ..., 𝑥ₚ (vad vi använder för att prediktera)
- **Modell**: 𝑦 = 𝑓(𝑥₁, 𝑥₂, ..., 𝑥ₚ) + 𝜖
- **Felterm**: 𝜖 (slumpmässig variation)

---

##  UTVÄRDERINGSMÅTT

### 1. **RMSE (Root Mean Square Error)** 

**VAD**: Kvadratroten av genomsnittliga kvadratiska felet
**VARFÖR**: Används mest - ger fel i samma enhet som target
**HUR**: Beräknas enligt formeln:

```
RMSE = √(1/n × Σ(y_true - y_pred)²)
```

**Exempel från boken:**

```python
# Importera funktionen för att beräkna RMSE från scikit-learn
from sklearn.metrics import root_mean_squared_error

# Skapa exempeldata med sanna värden (y_true) och predikterade värden (y_pred)
y_true = [3, -0.5, 2, 7]  # De faktiska värdena vi vill prediktera
y_pred = [2.5, 0.0, 2, 8]  # Våra modells prediktioner

# Beräkna RMSE mellan sanna och predikterade värden
rmse = root_mean_squared_error(y_true, y_pred)
# Resultat: 0.6123724356957945
```

### 2. **MSE (Mean Square Error)**

**VAD**: Genomsnittliga kvadratiska felet
**VARFÖR**: Billigare att beräkna än RMSE, samma rangordning
**HUR**: Beräknas enligt formeln:

```
MSE = 1/n × Σ(y_true - y_pred)²
```

**Exempel från boken:**

```python
# Importera funktionen för att beräkna MSE från scikit-learn
from sklearn.metrics import mean_squared_error

# Skapa exempeldata med sanna värden och predikterade värden
y_true = [3, -0.5, 2, 7]  # De faktiska värdena
y_pred = [2.5, 0.0, 2, 8]  # Våra modells prediktioner

# Beräkna MSE mellan sanna och predikterade värden
mse = mean_squared_error(y_true, y_pred)
# Resultat: 0.375
```

### 3. **MAE (Mean Absolute Error)**

**VAD**: Genomsnittliga absoluta felet
**VARFÖR**: När alla fel ska värderas linjärt
**HUR**: Beräknas enligt formeln:

```
MAE = 1/n × Σ|y_true - y_pred|
```

**Exempel från boken:**

```python
# Importera funktionen för att beräkna MAE från scikit-learn
from sklearn.metrics import mean_absolute_error

# Skapa exempeldata med sanna värden och predikterade värden
y_true = [3, -0.5, 2, 7]  # De faktiska värdena
y_pred = [2.5, 0.0, 2, 8]  # Våra modells prediktioner

# Beräkna MAE mellan sanna och predikterade värden
mae = mean_absolute_error(y_true, y_pred)
# Resultat: 0.5
```

###  **RMSE vs MAE - Viktig skillnad från boken:**

**Exempel från boken som visar skillnaden:**

```python
# Skapa två olika modeller med olika felprofil
y_true = [100, 80]  # Sanna värden

# Modell 1: Gör små fel på båda observationerna
y_pred_model_1 = [90, 75]  # Fel på 10 och 5

# Modell 2: Gör ett stort fel på en observation
y_pred_model_2 = [90, 15]  # Fel på 10 och 65

# Beräkna MAE för båda modellerna
MAE1 = mean_absolute_error(y_true, y_pred_model_1)  # 7.5
MAE2 = mean_absolute_error(y_true, y_pred_model_2)  # 37.5

# Beräkna RMSE för båda modellerna
RMSE1 = root_mean_squared_error(y_true, y_pred_model_1)  # 7.9
RMSE2 = root_mean_squared_error(y_true, y_pred_model_2)  # 46.5
```

**Analys:**

- **MAE**: Ökar linjärt (7.5  37.5, faktor 5)
- **RMSE**: Ökar icke-linjärt (7.9  46.5, faktor 5.89)
- **Slutsats**: RMSE straffar stora fel hårdare än MAE

---

##  REGRESSIONSMODELLER

### 1. **Linjär Regression** 

#### **Enkel linjär regression**

**VAD**: Predikterar 𝑦 baserat på en variabel 𝑥
**VARFÖR**: Grundmodellen för alla regressionsproblem
**HUR**: 𝑦 = θ₀ + θ₁𝑥

**Exempel från boken:**

```python
# Importera nödvändiga bibliotek
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Skapa exempeldata från boken
x = np.array([6, 7, 8, -4, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([25, 40, 30, 40, 28, 35, 12, 20, 15, 3, 8, 18, 5, 22, 26, 33])

# Skapa och träna en linjär regressionsmodell
lin_reg = LinearRegression()  # Skapa modellobjektet
lin_reg.fit(x, y)  # Träna modellen på datan

# Skriv ut modellens parametrar
print("Intercept:", lin_reg.intercept_)  # 22.18 - skärningspunkt med y-axeln
print("Slope:", lin_reg.coef_)          # [0.47] - lutningen
```

#### **Multipel linjär regression**

**VAD**: Predikterar 𝑦 baserat på flera variabler
**VARFÖR**: Kan hantera komplexare samband
**HUR**: 𝑦 = θ₀ + θ₁𝑥₁ + θ₂𝑥₂ + ... + θₚ𝑥ₚ

**Matrisform från boken:**

```
ŷ = Xθ̂
```

### 2. **Polynomregression** 

**VAD**: Linjär regression med högre potenser
**VARFÖR**: Kan fånga icke-linjära samband
**HUR**: 𝑦 = θ₀ + θ₁𝑥 + θ₂𝑥² + θ₃𝑥³ + ...

**Exempel från boken:**

```python
# Importera nödvändiga funktioner
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Skapa polynomtermer av grad 2 från ursprungliga data
poly = PolynomialFeatures(degree=2, include_bias=False)  # Skapa transformer
x_poly = poly.fit_transform(x)  # Transformera data till polynomtermer

# Träna en linjär regression på polynomtermerna
poly_reg = LinearRegression()  # Skapa linjär regressionsmodell
poly_reg.fit(x_poly, y)  # Träna modellen
```

**Viktigt från boken**: Polynomregression är fortfarande en linjär regression eftersom den är linjär med avseende på parametrarna θᵢ.

### 3. **Ridge Regression (L2-regularisering)** 

**VAD**: Linjär regression med straff för stora koefficienter
**VARFÖR**: Minskar överanpassning genom att krympa koefficienter
**HUR**: Kostnadsfunktion blir:

```
J(θ) = MSE(θ) + α × (1/2) × Σθᵢ²
```

**Exempel från boken:**

```python
# Importera Ridge regression och GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Definiera hyperparametrar att testa
hyperparams = {'alpha': [0.1, 1, 10]}  # Regulariseringsstyrka

# Skapa Ridge regression modell
ridge = Ridge()  # Skapa modellobjektet

# Skapa GridSearchCV för att hitta bästa alpha
ridge_gs = GridSearchCV(estimator=ridge, param_grid=hyperparams,
                       scoring='neg_mean_squared_error', cv=5)
ridge_gs.fit(X_train, y_train)  # Träna modellen

# Skriv ut resultaten
print("Optimized alpha:", ridge_gs.best_estimator_)  # Bästa alpha-värdet
print("Coefficients:", np.round(ridge_gs.best_estimator_.coef_, 5))  # Koefficienter
```

**Effekt av α från boken:**

- **Lågt α**: Mindre regularisering, koefficienter nära linjär regression
- **Högt α**: Starkare regularisering, koefficienter krymps mer

### 4. **Lasso Regression (L1-regularisering)** 

**VAD**: Linjär regression med straff för absoluta koefficienter
**VARFÖR**: Automatisk variabelselektion genom att sätta vissa koefficienter till 0
**HUR**: Kostnadsfunktion blir:

```
J(θ) = MSE(θ) + α × Σ|θᵢ|
```

**Exempel från boken:**

```python
# Importera Lasso regression
from sklearn.linear_model import Lasso

# Skapa Lasso modell med stort alpha för demonstration
lasso_reg = Lasso(alpha=9)  # Stort alpha för att visa effekten
lasso_reg.fit(X_train, y_train)  # Träna modellen

# Skriv ut koefficienterna
print("Coefficients:", np.round(lasso_reg.coef_, 2))
# Resultat: [29.78, 0., 80.6] - andra koefficienten satt till 0
```

**Intuition från boken**: Tänk dig en budget för koefficienter. Lasso sparar budgeten till de viktigaste variablerna och sätter resten till 0.

### 5. **Elastic Net** 

**VAD**: Kombination av Ridge och Lasso
**VARFÖR**: Bästa av båda världarna
**HUR**: Kostnadsfunktion blir:

```
J(θ) = MSE(θ) + r×α×Σ|θᵢ| + (1-r)/2×α×Σθᵢ²
```

**Exempel från boken:**

```python
# Importera Elastic Net
from sklearn.linear_model import ElasticNet

# Definiera hyperparametrar för grid search
hyperparams = {
    'alpha': [0.01, 0.1, 1.0, 10.0],  # Regulariseringsstyrka
    'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0]  # Mix ratio mellan L1 och L2
}

# Skapa Elastic Net modell
elastic_net = ElasticNet()  # Skapa modellobjektet

# Skapa GridSearchCV för hyperparameteroptimering
elastic_net_gs = GridSearchCV(estimator=elastic_net, param_grid=hyperparams,
                             scoring='neg_mean_squared_error', cv=5)
elastic_net_gs.fit(X_train, y_train)  # Träna modellen
```

### 6. **Support Vector Machines (SVM)** 

**VAD**: Skapar en "väg" runt data med given bredd
**VARFÖR**: Robust, bra för små dataset
**HUR**: Skapar en väg med bredd ε som innehåller flest observationer

**Viktiga koncept från boken:**

- **ε-insensitive**: Observationer innanför vägen påverkar inte träningen
- **Kernel trick**: Ger effekten av transformerade variabler utan att faktiskt lägga till dem

**Exempel från boken:**

```python
# Importera nödvändiga funktioner
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Skapa pipeline med standardisering och SVM
svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardisera data först
    ('svr', LinearSVR())  # Sedan SVM
])

# Definiera hyperparametrar för epsilon
hyperparams = {'svr__epsilon': [0.01, 0.1, 1.0]}  # Bredden på vägen

# Skapa GridSearchCV för hyperparameteroptimering
svr_grid_search = GridSearchCV(estimator=svr_pipeline, param_grid=hyperparams,
                              scoring='neg_mean_squared_error')
svr_grid_search.fit(X_train, y_train)  # Träna modellen
```

### 7. **Beslutsträd** 

**VAD**: Trädstruktur med ja/nej-frågor
**VARFÖR**: Tolkbar, hanterar icke-linjära samband
**HUR**: Delar data upp i regioner med ja/nej-frågor

**Exempel från boken:**

```python
# Importera Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

# Skapa beslutsträd modell
dec_tree = DecisionTreeRegressor(random_state=42)  # Skapa modellobjektet

# Definiera hyperparametrar för grid search
hyperparams = {
    'max_depth': [2, 3, 5, 10],  # Maximalt djup av trädet
    'min_samples_split': [2, 5, 10]  # Minsta antal samples för att dela en nod
}

# Skapa GridSearchCV för hyperparameteroptimering
dec_tree_gs = GridSearchCV(estimator=dec_tree, param_grid=hyperparams,
                          scoring='neg_mean_squared_error')
dec_tree_gs.fit(X_train, y_train)  # Träna modellen
```

**Viktiga hyperparametrar från boken:**

- **max_depth**: Trädets maximala djup
- **min_samples_split**: Minsta antal observationer för att dela en nod
- **min_samples_leaf**: Minsta antal observationer i en lövnod

**White box vs Black box från boken:**

- **Beslutsträd**: White box - lätt att förstå varför en prediktion görs
- **Random Forest/Neurala nätverk**: Black box - svårt att intuitivt förklara

### 8. **Ensemble Learning** 

#### **Voting Regression**

**VAD**: Flera modeller röstar, tar medelvärde
**VARFÖR**: Kombinerar styrkorna hos olika modeller
**HUR**: Träna flera modeller och ta medelvärde av prediktionerna

**Exempel från boken:**

```python
# Importera nödvändiga modeller och funktioner
from sklearn.ensemble import VotingRegressor

# Skapa tre olika modeller för ensemble
m1 = LinearRegression()  # Linjär regression
m2 = DecisionTreeRegressor(max_depth=3, random_state=2)  # Beslutsträd
m3 = LinearSVR(random_state=2)  # Support Vector Regression

# Träna varje modell individuellt
m1.fit(X_train, y_train)  # Träna linjär regression
m2.fit(X_train, y_train)  # Träna beslutsträd
m3.fit(X_train, y_train)  # Träna SVM

# Gör prediktioner med varje modell
pred1 = m1.predict(X_test)  # Prediktioner från linjär regression
pred2 = m2.predict(X_test)  # Prediktioner från beslutsträd
pred3 = m3.predict(X_test)  # Prediktioner från SVM

# Beräkna ensemble prediktion som medelvärde
avg_pred = (pred1 + pred2 + pred3) / 3  # Manuell ensemble

# Använd VotingRegressor för automatisk ensemble
voting_regressor = VotingRegressor([
    ('lin_reg', m1), ('dec_tree', m2), ('svm', m3)  # Namnge modellerna
])
voting_regressor.fit(X_train, y_train)  # Träna ensemble
```

#### **Bagging (Bootstrap Aggregating)**

**VAD**: Skapar flera dataset med återläggning, tränar modeller
**VARFÖR**: Minskar varians, stabilare prediktioner
**HUR**:

1. Skapa nya dataset med slumpmässigt urval med återläggning
2. Träna modell på varje dataset
3. Kombinera prediktioner genom medelvärde

**Exempel från boken:**

```python
# Importera BaggingRegressor
from sklearn.ensemble import BaggingRegressor

# Skapa bagging ensemble med linjär regression som basmodell
bagging_reg = BaggingRegressor(
    estimator=LinearRegression(),  # Basmodell
    n_estimators=15  # Antal modeller i ensemble
)
bagging_reg.fit(X_train, y_train)  # Träna ensemble
```

#### **Pasting**

**VAD**: Som bagging men utan återläggning
**VARFÖR**: Mindre överlapp mellan dataset
**HUR**: Sätt `bootstrap=False` i BaggingRegressor

### 9. **Random Forest** 

**VAD**: Ensemble av beslutsträd
**VARFÖR**: Mycket robust, bra prestanda
**HUR**: Många beslutsträd med bagging + slumpmässig variabelselektion

**Exempel från boken:**

```python
# Importera Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Skapa Random Forest modell
random_forest = RandomForestRegressor(random_state=42)  # Skapa modellobjektet

# Definiera hyperparametrar för grid search
hyperparams = {
    'n_estimators': [10, 50, 100, 150],  # Antal träd i skogen
    'max_depth': [2, 3, 5, 10],  # Maximalt djup för varje träd
    'min_samples_split': [2, 5, 10]  # Minsta antal samples för att dela en nod
}

# Skapa GridSearchCV för hyperparameteroptimering
random_forest_gs = GridSearchCV(estimator=random_forest, param_grid=hyperparams,
                               scoring='neg_mean_squared_error')
random_forest_gs.fit(X_train, y_train)  # Träna modellen
```

---

##  VIKTIGA KONCEPT

### **Gradient Descent** ⬇

**VAD**: Optimeringsalgoritm som hittar minimum
**VARFÖR**: Central algoritm inom ML för att hitta optimala parametrar
**HUR**: Gå i riktningen med negativ lutning tills lutningen är 0

**Varianter från boken:**

1. **Batch Gradient Descent**: Använder hela träningsdatan för varje iteration
2. **Stochastic Gradient Descent**: Väljer slumpmässigt en observation i taget
3. **Mini-batch Gradient Descent**: Väljer slumpmässigt en mindre batch av data

**Viktigt från boken**: Standardisering av data gör det lättare för gradient descent att nå optimum.

### **Bias-Variance Trade-off** 

**VAD**: Viktigt koncept för modellval
**VARFÖR**: Förklarar varför mer komplexa modeller inte alltid är bättre
**HUR**: Felet kan delas in i tre delar:

1. **Bias**: Fel från felaktiga modellantaganden (underanpassning)
2. **Variance**: Fel från känslighet för fluktuationer i träningsdata (överanpassning)
3. **Irreducerbart fel**: Fel från slumpmässighet i datan

**Exempel från boken:**

- **Låg bias, hög variance**: Polynomregression med hög grad
- **Hög bias, låg variance**: Enkel linjär regression
- **Mål**: Hitta balansen mellan bias och variance

### **Regularisering** 

**VAD**: Tekniker för att förhindra överanpassning
**VARFÖR**: Minskar modellens komplexitet
**HUR**: Höjer bias, sänker variance

**Metoder från boken:**

- **Ridge (L2)**: Krymper koefficienter mot 0
- **Lasso (L1)**: Sätter vissa koefficienter till exakt 0
- **Elastic Net**: Kombination av båda

---

##  PRAKTISKA TIPS FRÅN BOKEN

### **Modellval** 

1. **Börja enkelt**: Linjär regression
2. **Testa komplexare**: Polynomregression, beslutsträd
3. **Ensemble**: Random Forest, Voting
4. **Regularisera**: Ridge, Lasso om överanpassning

### **Hyperparameteroptimering** 

1. **Grid Search**: Testa alla kombinationer
2. **Korsvalidering**: Robust utvärdering
3. **Exempel från boken**: Använd `GridSearchCV` med `scoring='neg_mean_squared_error'`

### **Feature Engineering** 

1. **Standardisering**: Viktigt för många modeller (särskilt SVM)
2. **Polynomtermer**: För icke-linjära samband
3. **Interaktioner**: Kombinera variabler

### **Utvärdering** 

1. **RMSE**: Standardmått
2. **Korsvalidering**: Robust bedömning
3. **Visualisering**: Förstå modellens beteende

---

##  NYCKELPOÄNG FRÅN BOKEN

1. **Regression** = Prediktera kontinuerliga värden
2. **RMSE** = Standardmått för regression
3. **Linjär regression** = Bra utgångspunkt
4. **Regularisering** = Förhindra överanpassning
5. **Ensemble** = Kombinera modeller för bättre prestanda
6. **Bias-Variance** = Hitta rätt komplexitet
7. **Gradient Descent** = Hitta optimala parametrar
8. **White box vs Black box** = Tolkbarhet vs prestanda

---

##  RELATIONER MELLAN MODELLER

```
Linjär Regression
    ↓ (lägg till polynomtermer)
Polynomregression
    ↓ (lägg till regularisering)
Ridge/Lasso/Elastic Net
    ↓ (kombinera flera modeller)
Ensemble Learning
    ↓ (specifikt för träd)
Random Forest
```

---

##  PRAKTISKA EXEMPEL FRÅN BOKEN

### **Syntetiskt dataset från boken:**

```python
# Importera funktioner för att skapa syntetiskt data
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Skapa syntetiskt dataset med 5000 observationer och 3 features
X, y = make_regression(n_samples=5000, n_features=3, noise=5, random_state=42)

# Dela upp data i träning och test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### **Jämförelse av modeller från boken:**

```python
# Träna linjär regression
lin_reg = LinearRegression()  # Skapa modellobjektet
lin_reg.fit(X_train, y_train)  # Träna modellen
y_pred_lin_reg = lin_reg.predict(X_test)  # Gör prediktioner
RMSE_lin_reg = root_mean_squared_error(y_test, y_pred_lin_reg)  # Beräkna RMSE

# Skapa pipeline för polynomregression
poly_reg_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # Skapa polynomtermer
    ('polyreg', LinearRegression())  # Linjär regression på polynomtermer
])
poly_reg_pipeline.fit(X_train, y_train)  # Träna pipeline
y_pred_poly_reg = poly_reg_pipeline.predict(X_test)  # Gör prediktioner
RMSE_poly_reg = root_mean_squared_error(y_test, y_pred_poly_reg)  # Beräkna RMSE

# Jämför resultaten
print("RMSE Linear Regression:", RMSE_lin_reg)
print("RMSE Polynomial Regression:", RMSE_poly_reg)
```

---

##  VISUALISERINGAR OCH GRAFER FRÅN BOKEN

### **Linjär Regression Visualisering**

- **Figur 3.1**: Enkel linjär regression med data, predikterad linje och residualer
- **Figur 3.2**: Multipel linjär regression som ett plan i 3D

### **Gradient Descent Visualiseringar**

- **Figur 3.3**: Gradient descent algoritmen med steg mot minimum
- **Figur 3.4**: Risk för lokala minimum
- **Figur 3.5**: För stora steg kan leda till hoppande
- **Figur 3.6**: Effekt av olika learning rates
- **Figur 3.7**: Tre varianter av gradient descent
- **Figur 3.8**: Effekt av feature scaling

### **Bias-Variance Trade-off**

- **Figur 3.9**: Jämförelse mellan enkel linjär regression och polynomregression
- **Figur 3.10**: Effekt av α i Ridge regression
- **Figur 3.11**: Effekt av α i Lasso regression

### **Beslutsträd**

- **Figur 3.13**: Visualisering av beslutsträd med noder och frågor
- **Figur 3.14**: Beslutsträd som delar upp datarummet
- **Figur 3.15**: Överanpassat vs regulariserat träd

### **Ensemble Learning**

- **Figur 3.16**: Schematisk bild över voting regression
- **Figur 3.17**: Bagging och pasting process

---

##  AVANCERADE KONCEPT FRÅN BOKEN

### **Kernel Trick (SVM)**

- Ger effekten av transformerade variabler utan att faktiskt lägga till dem
- Påverkar inte träningstiden
- Olika kernels: 'linear', 'poly', 'rbf', 'sigmoid'

### **Out-of-Bag Observations (Bagging)**

- Cirka 37% av observationerna aldrig dragna i bagging
- Kan användas för utvärdering
- Styrs med `oob_score=True`

### **Random Subspaces vs Random Patches**

- **Random Subspaces**: Hela dataset, slumpmässig andel features
- **Random Patches**: Slumpmässig andel både data och features
- Användbart för högdimensionella dataset

---

##  VIKTIGA BEGREPP OCH DEFINITIONER

### **Överanpassning (Overfitting)**

- **VAD**: Modellen lär sig träningsdata för väl och generaliserar dåligt till ny data
- **VARFÖR**: Högre komplexitet leder till att modellen "memorerar" träningsdata
- **HUR**: Använd regularisering, minska modellkomplexitet, använd mer träningsdata

### **Underanpassning (Underfitting)**

- **VAD**: Modellen är för enkel och kan inte fånga mönster i datan
- **VARFÖR**: För låg komplexitet eller felaktiga antaganden
- **HUR**: Öka modellkomplexitet, lägg till features, minska regularisering

### **Korsvalidering (Cross-validation)**

- **VAD**: Teknik för att utvärdera modeller på begränsad data
- **VARFÖR**: Ger mer robust uppskattning av modellprestanda
- **HUR**: Dela data i k fold, träna på k-1 fold, testa på 1 fold, upprepa

### **Hyperparametrar**

- **VAD**: Parametrar som sätts innan träning (inte lärda av modellen)
- **VARFÖR**: Kontrollerar modellens beteende och komplexitet
- **EXEMPEL**: α i Ridge/Lasso, max_depth i beslutsträd, n_estimators i Random Forest

### **Feature Engineering**

- **VAD**: Process att skapa, transformera eller välja features
- **VARFÖR**: Förbättra modellprestanda genom bättre representation av data
- **EXEMPEL**: Standardisering, polynomtermer, interaktioner, feature selection

### **Pipeline**

- **VAD**: Sekvens av transformer och estimator
- **VARFÖR**: Automatisera preprocessing och modellering
- **EXEMPEL**: Standardisering  Polynomtermer  Linjär regression

### **Residualer**

- **VAD**: Skillnaden mellan sanna och predikterade värden
- **VARFÖR**: Visar modellens fel och kan användas för diagnostik
- **FORMEL**: eᵢ = yᵢ - ŷᵢ

### **R² (R-squared)**

- **VAD**: Mått på hur mycket av variansen i target som förklaras av modellen
- **VARFÖR**: Tolkar modellens anpassning (0-1, högre är bättre)
- **FORMEL**: R² = 1 - (SS_res / SS_tot)

### **Feature Scaling**

- **VAD**: Normalisera features till samma skala
- **VARFÖR**: Viktigt för många algoritmer (SVM, gradient descent)
- **METODER**: StandardScaler (z-score), MinMaxScaler (0-1)

### **Ensemble Learning**

- **VAD**: Kombinera flera modeller för bättre prestanda
- **VARFÖR**: Minskar varians, ökar robusthet
- **METODER**: Voting, Bagging, Boosting, Stacking

### **Regularisering**

- **VAD**: Tekniker för att förhindra överanpassning
- **VARFÖR**: Minska modellkomplexitet, förbättra generalisering
- **METODER**: L1 (Lasso), L2 (Ridge), Elastic Net

---

_Denna omfattande sammanfattning ger dig en komplett förståelse för regressionsmodeller baserat på bokens innehåll!_ 
