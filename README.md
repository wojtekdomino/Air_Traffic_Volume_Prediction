# Predykcja Natężenia Ruchu Lotniczego

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

## Przykładowe Wyniki

Po uruchomieniu projektu, w pliku `models/model_comparison.csv` znajdziesz tabelę porównawczą:

| Model              | RMSE    | MAE     | R²      | Rozmiar (MB) |
|--------------------|---------|---------|---------|--------------|
| LightGBM           | ~XXX    | ~XXX    | ~0.XX   | ~X.XX        |
| MLP FP32           | ~XXX    | ~XXX    | ~0.XX   | ~X.XX        |
| MLP Pruned         | ~XXX    | ~XXX    | ~0.XX   | ~X.XX        |
| MLP Quantized INT8 | ~XXX    | ~XXX    | ~0.XX   | ~X.XX        |

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

---

## Licencja

Projekt edukacyjny - użytek akademicki

### Categorical Encodings
- Encoded `APT_ICAO` (airport)
- Encoded `STATE_NAME` (country)
- Encoded `SEASON`

## Models

### 1. LightGBM Regressor (Baseline)
- **Type**: Gradient boosting trees
- **Hyperparameters**: 
  - Learning rate: 0.05
  - Num leaves: 31
  - Early stopping: 50 rounds
- **Output**: Feature importance plot

### 2. MLP (Multi-Layer Perceptron)
- **Architecture**: Input → [128, 64, 32] → Output(1)
- **Activation**: ReLU
- **Optimizer**: Adam (lr=0.001)
- **Loss**: MSE
- **Epochs**: 50
- **Dropout**: 0.2

### 3. Pruned MLP
- **Method**: Structured L-n pruning
- **Pruning amount**: 30% of neurons
- **Fine-tuning**: 10 epochs at lr=0.0001

### 4. Quantized MLP
- **Method**: Dynamic quantization
- **Type**: INT8 (from FP32)
- **Backend**: fbgemm (CPU)
- **Target layers**: Linear layers

## Evaluation Metrics

Each model is evaluated on:

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of determination)
- **Model Size** (MB)
- **Inference Time** (ms/sample)

## Visualizations

The evaluation script generates:

1. **Comparison Table** - Summary of all metrics
2. **Predictions vs Actual** - Scatter plots for each model
3. **Residuals Distribution** - Histogram of prediction errors
4. **Metrics Comparison** - Bar charts comparing RMSE, R², size, and speed

All plots are saved in the `models/` directory.

## Model Files

After training, the following files are saved:

- `lightgbm_model.txt` - LightGBM model (~1-5 MB)
- `mlp_fp32.pt` - Full precision MLP (~0.5 MB)
- `mlp_pruned.pt` - Pruned MLP (~0.5 MB, 30% sparse)
- `mlp_int8.pt` - Quantized MLP (~0.1-0.2 MB, 4x smaller)
- `mlp_scaler.pkl` - StandardScaler for MLP inputs
- `evaluation_results.csv` - Detailed comparison table

## Expected Results

Typical performance (will vary based on your dataset):

| Model | RMSE | R² | Size | Speed |
|-------|------|-----|------|-------|
| LightGBM | ~XXX | ~0.XX | ~X MB | ~0.00X ms |
| MLP FP32 | ~XXX | ~0.XX | ~X MB | ~0.00X ms |
| MLP Pruned | ~XXX | ~0.XX | ~X MB | ~0.00X ms |
| MLP INT8 | ~XXX | ~0.XX | ~X MB | ~0.00X ms |

**Compression Benefits**:
- **Size reduction**: ~75% (FP32 → INT8)
- **Speed improvement**: ~2-4x faster inference
- **Accuracy retention**: Minimal loss (<2% RMSE increase)

## Methodology

### Data Preprocessing
1. Load CSV data
2. Remove duplicates
3. Handle missing values
4. Keep relevant columns

### Feature Engineering
1. Create time-based features
2. Add seasonal indicators
3. Generate lag features per airport
4. Encode categorical variables

### Model Training
1. **LightGBM**: Direct training with early stopping
2. **MLP**: 
   - Standardize features
   - Train with MSE loss
   - Save best model

### Model Compression
1. **Pruning**:
   - Apply L-n structured pruning (30%)
   - Fine-tune for 10 epochs
   - Remove pruning masks
2. **Quantization**:
   - Dynamic quantization FP32 → INT8
   - Evaluate accuracy retention
   - Measure size/speed improvements

## Notes

- **Data Quality**: Ensure your CSV has no major data quality issues
- **Memory**: Large datasets may require chunking or sampling
- **GPU**: MLP training can use CUDA if available
- **Quantization**: INT8 models run best on CPU (fbgemm backend)

## Contributing

This project was created as a demonstration of:
- Machine learning regression pipeline
- Model compression techniques
- PyTorch neural network implementation
- LightGBM gradient boosting
- Comprehensive model evaluation

## License

This project is open source and available for educational purposes.

## Acknowledgments

- **Dataset**: European Aviation Safety Agency (EASA) / Eurocontrol
- **Libraries**: PyTorch, LightGBM, scikit-learn, pandas
- **Model Compression**: PyTorch quantization and pruning APIs

---

**Author**: Wojciech Domino & Mateusz Maj 
**Date**: November 2025  
**Purpose**: Air traffic volume prediction with model compression
