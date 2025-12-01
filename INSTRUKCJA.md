# INSTRUKCJA URUCHOMIENIA PROJEKTU

## Krok po kroku

### Krok 1: Zainstaluj wymagane biblioteki

Otwórz terminal w folderze projektu i wykonaj:

```bash
pip install -r requirements.txt
```

**Co zostanie zainstalowane:**
- pandas, numpy - analiza danych
- scikit-learn - preprocessing
- lightgbm - model gradient boosting
- torch - sieci neuronowe
- matplotlib, seaborn - wizualizacje

---

### Krok 2: Sprawdź dane

Upewnij się, że plik `data/european_flights.csv` istnieje.

---

### Krok 3: Uruchom projekt

Wystarczy jedno polecenie:

```bash
python main.py
```

**Projekt automatycznie:**
1. Wczyta i oczyści dane
2. Stworzy cechy (feature engineering)
3. Wytrenuje model LightGBM
4. Wytrenuje sieć neuronową MLP
5. Skompresuje modele (pruning + quantization)
6. Porówna wszystkie modele

---

### Krok 4: Zobacz wyniki

Po zakończeniu, sprawdź folder `models/`:
- `lightgbm_model.txt` - wytrenowany model LightGBM
- `mlp_fp32.pt` - sieć neuronowa (full precision)
- `mlp_pruned.pt` - sieć po przycinaniu
- `mlp_int8.pt` - sieć po kwantyzacji
- `model_comparison.csv` - **TABELA PORÓWNAWCZA WSZYSTKICH MODELI**

---

## Rozwiązywanie problemów

### Problem: Brak modułu lightgbm lub torch

**Rozwiązanie:**
```bash
pip install lightgbm torch
```

### Problem: Brak pliku danych

**Rozwiązanie:**  
Umieść plik `european_flights.csv` w folderze `data/`

### Problem: Błąd pamięci

**Rozwiązanie:**  
Zmniejsz `BATCH_SIZE` w pliku `main.py` (linia ~40):
```python
BATCH_SIZE = 128  # zamiast 256
```

---

## Analiza kodu

Plik `main.py` jest podzielony na **8 sekcji**:

1. **Import i konfiguracja** (linie 1-60)
2. **Preprocessing** (linie 61-150)
3. **Feature engineering** (linie 151-350)
4. **LightGBM** (linie 351-550)
5. **MLP** (linie 551-850)
6. **Kompresja** (linie 851-1050)
7. **Porównanie** (linie 1051-1200)
8. **Main** (linie 1201-koniec)

Każda funkcja ma **szczegółowe komentarze** wyjaśniające:
- Co robi
- Dlaczego jest potrzebna
- Jak działa

---

## Wskazówki

### Chcesz zmienić architekturę sieci?

Edytuj klasę `MLPRegressor` (około linia 600):
```python
hidden_sizes=[128, 64, 32]  # Możesz zmienić na [256, 128, 64]
```

### Chcesz więcej/mniej epok treningu?

Zmień stałą `EPOCHS` (linia ~40):
```python
EPOCHS = 50  # zamiast 100
```

### Chcesz bardziej agresywny pruning?

Zmień `PRUNING_AMOUNT` (linia ~45):
```python
PRUNING_AMOUNT = 0.5  # 50% neuronów zamiast 30%
```

---

## Struktura akademicka

Projekt został zaprojektowany jako materiał edukacyjny:

- Jeden plik - łatwa nawigacja
- Szczegółowe komentarze w języku polskim
- Logiczny podział na sekcje
- Wyjaśnienie każdej techniki
- Pełny pipeline od danych do wyników  

**Idealny do:**
- Nauki machine learning
- Projektów studenckich
- Zrozumienia kompresji modeli
- Porównywania różnych podejść

---

## Co można się nauczyć z tego projektu?

1. **Preprocessing danych** - czyszczenie, agregacja
2. **Feature engineering** - tworzenie cech czasowych, sezonowych, lag features
3. **Gradient boosting** - LightGBM
4. **Sieci neuronowe** - PyTorch, MLP, trenowanie
5. **Kompresja modeli** - pruning, quantization
6. **Ewaluacja** - metryki, porównywanie modeli

---

## Wymagania

- Python 3.8+
- ~500 MB wolnego miejsca na dysku (modele + dane)
- Zalecane: 8 GB RAM

---

## Szybkie przypomnienie

```bash
# 1. Zainstaluj
pip install -r requirements.txt

# 2. Uruchom
python main.py

# 3. Zobacz wyniki
cat models/model_comparison.csv
```

**To wszystko! Projekt powinien działać out-of-the-box.**

---

Powodzenia!
