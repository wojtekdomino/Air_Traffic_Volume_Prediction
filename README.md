# Predykcja Natężenia Ruchu Lotniczego

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-9ACD32?logo=lightgbm&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-blueviolet)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-MLP-orange)
![Model Compression](https://img.shields.io/badge/Model%20Compression-Pruning%20%26%20Quantization-green)

[English version below](#english-version)

## Opis Projektu

Projekt akademicki z zakresu uczenia maszynowego i uczenia głębokiego, którego celem jest **predykcja liczby operacji lotniczych IFR (Instrument Flight Rules)** na lotniskach europejskich na podstawie danych historycznych.

### Cel Projektu
Porównanie skuteczności różnych modeli uczenia maszynowego w zadaniu regresji:
- **LightGBM** - model bazowy (gradient boosting)
- **MLP (Multi-Layer Perceptron)** - sieć neuronowa w PyTorch
- **Techniki kompresji modeli**: pruning i quantization

### Dataset
**European Flights Dataset** - miesięczne dane agregowane o operacjach lotniczych na lotniskach europejskich.

**Zmienna docelowa**: `FLT_TOT_1` (całkowita liczba operacji IFR)

**Cechy wejściowe**:
- `YEAR`, `MONTH_NUM` - informacje czasowe
- `APT_ICAO` - kod ICAO lotniska
- `STATE_NAME` - kraj
- `FLT_DEP_1`, `FLT_ARR_1` - liczba odlotów i przylotów

---

## Struktura Projektu

```
Air_Traffic_Volume_Prediction/
├── main.py                           # GŁÓWNY PLIK - cały kod projektu
├── data/
│   └── european_flights.csv          # Dataset
├── models/                            # Wytrenowane modele (generowane automatycznie)
│   ├── lightgbm_model.txt
│   ├── mlp_fp32.pt
│   ├── mlp_pruned.pt
│   ├── mlp_int8.pt
│   ├── mlp_scaler.pkl
│   └── model_comparison.csv          # Tabela porównawcza
├── requirements.txt                   # Zależności
└── README.md                         # Ten plik
```

UWAGA: Cały kod projektu został skonsolidowany w jednym pliku `main.py` dla łatwiejszej nawigacji i zrozumienia struktury projektu.

---

## Szybki Start

### 1. Instalacja Zależności

```bash
pip install -r requirements.txt
```

### 2. Przygotowanie Danych

Upewnij się, że plik `european_flights.csv` znajduje się w folderze `data/`.

### 3. Uruchomienie Projektu

**Cały projekt uruchamia się jednym poleceniem:**

```bash
python main.py
```

### Czego Się Spodziewać?

Po uruchomieniu, program automatycznie wykonuje:

1. Wczytanie i czyszczenie danych
2. Inżynierię cech (time features, seasonal features, lag features)
3. Trenowanie modelu LightGBM
4. Trenowanie sieci neuronowej MLP
5. Kompresję modeli (pruning i quantization)
6. Porównanie wszystkich modeli i zapisanie wyników

**Wyniki** zapisywane są w folderze `models/`:
- Wytrenowane modele (`.pt`, `.txt`)
- Tabela porównawcza (`model_comparison.csv`)

---

## Metodologia

### Przygotowanie Danych

1. **Czyszczenie**: usunięcie duplikatów i brakujących wartości
2. **Selekcja cech**: wybór istotnych kolumn

### Inżynieria Cech

**Cechy czasowe:**
- `YEAR_TREND` - znormalizowany trend roczny
- `MONTH_SIN`, `MONTH_COS` - cykliczne kodowanie miesiąca

**Cechy sezonowe:**
- `SEASON` - pora roku (Winter/Spring/Summer/Fall)
- `IS_SUMMER`, `IS_WINTER` - flagi binarne

**Cechy opóźnione (lag features):**
- `lag_1` - wartość z poprzedniego miesiąca
- `lag_3` - średnia krocząca z 3 miesięcy

**Kodowanie kategorii:**
- Label Encoding dla: `APT_ICAO`, `STATE_NAME`, `SEASON`

### Modele

#### 1. LightGBM (Baseline)
- Gradient Boosting Decision Trees
- Szybki i efektywny dla tabulacyjnych danych
- Early stopping dla uniknięcia przeuczenia

#### 2. MLP (Multi-Layer Perceptron)
- Architektura: Input → [128, 64, 32] → Output
- Funkcja aktywacji: ReLU
- Regularyzacja: Dropout (20%)
- Optimizer: Adam
- Loss: MSE (Mean Squared Error)

#### 3. MLP z Pruningiem
- Structured pruning (30% neuronów)
- Redukcja rozmiaru modelu przy minimalnej utracie accuracy

#### 4. MLP z Kwantyzacją
- Dynamic quantization FP32 → INT8
- ~4x redukcja rozmiaru modelu
- Przyspieszenie inferencingu

### Metryki Ewaluacji

- **RMSE** (Root Mean Squared Error) - pierwiastek błędu średniokwadratowego
- **MAE** (Mean Absolute Error) - średni błąd bezwzględny
- **R²** (R-squared) - współczynnik determinacji
- **Rozmiar modelu** (MB)

---

## Wyniki

Po uruchomieniu projektu, w pliku `models/model_comparison.csv` znajdziesz tabelę porównawczą:

| Model              | RMSE    | MAE     | R²      | Rozmiar (MB) |
|--------------------|---------|---------|---------|--------------|
| LightGBM           | 1.03    | 0.25    | 0.9999  | 0.56         |
| MLP FP32           | 13.21   | 10.89   | 0.9965  | 0.05         |
| MLP Pruned         | 127.19  | 58.92   | 0.6717  | 0.05         |
| MLP Quantized INT8 | 15.19   | 11.90   | 0.9953  | 0.02         |

**Wnioski:**
- **LightGBM** osiągnął najlepszą dokładność (R² = 0.9999)
- **MLP Quantized INT8** oferuje najlepszy kompromis - dobra accuracy (R² = 0.9953) przy najmniejszym rozmiarze (0.02 MB)
- **Pruning** znacząco obniża accuracy, ale utrzymuje mały rozmiar modelu
- **Kwantyzacja** redukuje rozmiar o ~60% przy minimalnej utracie accuracy

---

## Wymagania Techniczne

**Python**: 3.8+

**Biblioteki**:
- `pandas` - analiza danych
- `numpy` - operacje numeryczne
- `scikit-learn` - preprocessing, metryki
- `lightgbm` - model gradient boosting
- `torch` - sieci neuronowe
- `matplotlib`, `seaborn` - wizualizacje (opcjonalne)

---

## Struktura Kodu (main.py)

Kod podzielony jest na **8 logicznych sekcji**:

1. **Import bibliotek i konfiguracja** - stałe, ustawienia
2. **Preprocessing** - wczytywanie i czyszczenie danych
3. **Feature Engineering** - tworzenie cech
4. **Trenowanie LightGBM** - model bazowy
5. **Trenowanie MLP** - sieć neuronowa
6. **Kompresja modeli** - pruning i quantization
7. **Porównanie modeli** - ewaluacja i analiza
8. **Funkcja główna** - pipeline projektu

Każda funkcja ma **szczegółowe komentarze akademickie** wyjaśniające:
- Co robi funkcja
- Jakie parametry przyjmuje
- Co zwraca
- Dlaczego dana technika jest stosowana

---

## Wskazówki

### Dostosowanie Hiperparametrów

W pliku `main.py` w sekcji 1 znajdziesz stałe konfiguracyjne:

```python
RANDOM_STATE = 42          # Ziarno losowości
TEST_SIZE = 0.2           # Proporcja zbioru testowego
BATCH_SIZE = 256          # Batch size dla MLP
LEARNING_RATE = 0.001     # Learning rate
EPOCHS = 100              # Liczba epok
PRUNING_AMOUNT = 0.3      # Procent neuronów do przycięcia
```

### Eksperymentowanie

Możesz łatwo zmodyfikować:
- Architekturę MLP (sekcja 5)
- Hiperparametry LightGBM (sekcja 4)
- Stopień kompresji (sekcja 6)

---

## Literatura i Referencje

- **LightGBM**: [Ke et al., 2017 - LightGBM: A Highly Efficient Gradient Boosting Decision Tree]
- **Pruning**: [Han et al., 2015 - Learning both Weights and Connections for Efficient Neural Networks]
- **Quantization**: [Jacob et al., 2018 - Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference]

---

## Autor

Projekt akademicki - Machine Learning & Deep Learning

**Autorzy:** Wojciech Domino & Mateusz Maj

---

## Licencja

Projekt edukacyjny - użytek akademicki

---

## English Version

# Air Traffic Volume Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-9ACD32?logo=lightgbm&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-blueviolet)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-MLP-orange)
![Model Compression](https://img.shields.io/badge/Model%20Compression-Pruning%20%26%20Quantization-green)

## Project Description

Academic project in machine learning and deep learning focused on **predicting IFR (Instrument Flight Rules) operations** at European airports based on historical data.

### Project Goal
Compare the effectiveness of different machine learning models in regression tasks:
- **LightGBM** - baseline model (gradient boosting)
- **MLP (Multi-Layer Perceptron)** - PyTorch neural network
- **Model compression techniques**: pruning and quantization

### Dataset
**European Flights Dataset** - monthly aggregated data on flight operations at European airports.

**Target variable**: `FLT_TOT_1` (total IFR operations)

**Input features**:
- `YEAR`, `MONTH_NUM` - temporal information
- `APT_ICAO` - ICAO airport code
- `STATE_NAME` - country
- `FLT_DEP_1`, `FLT_ARR_1` - number of departures and arrivals

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Ensure `european_flights.csv` is in the `data/` folder.

### 3. Run the Project
```bash
python main.py
```

The project automatically:
1. Loads and cleans data
2. Performs feature engineering
3. Trains LightGBM model
4. Trains MLP neural network
5. Compresses models (pruning & quantization)
6. Compares all models and saves results

## Results

| Model              | RMSE    | MAE     | R²      | Size (MB) |
|--------------------|---------|---------|---------|-----------|
| LightGBM           | 1.03    | 0.25    | 0.9999  | 0.56      |
| MLP FP32           | 13.21   | 10.89   | 0.9965  | 0.05      |
| MLP Pruned         | 127.19  | 58.92   | 0.6717  | 0.05      |
| MLP Quantized INT8 | 15.19   | 11.90   | 0.9953  | 0.02      |

**Key findings:**
- **LightGBM** achieved the best accuracy (R² = 0.9999)
- **MLP Quantized INT8** offers the best trade-off - good accuracy (R² = 0.9953) with smallest size (0.02 MB)
- **Pruning** significantly reduces accuracy but maintains small model size
- **Quantization** reduces size by ~60% with minimal accuracy loss

## Technical Requirements

**Python**: 3.8+

**Libraries**:
- `pandas` - data analysis
- `numpy` - numerical operations
- `scikit-learn` - preprocessing, metrics
- `lightgbm` - gradient boosting model
- `torch` - neural networks
- `matplotlib`, `seaborn` - visualizations (optional)

## Code Structure (main.py)

The code is divided into **8 logical sections**:

1. **Imports and configuration** - constants, settings
2. **Preprocessing** - loading and cleaning data
3. **Feature engineering** - creating features
4. **LightGBM training** - baseline model
5. **MLP training** - neural network
6. **Model compression** - pruning and quantization
7. **Model comparison** - evaluation and analysis
8. **Main function** - project pipeline

Each function has **detailed academic comments** explaining:
- What the function does
- What parameters it accepts
- What it returns
- Why the technique is used

## Authors

**Wojciech Domino & Mateusz Maj**

Academic project - Machine Learning & Deep Learning

## License

Educational project - academic use
