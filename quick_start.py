#!/usr/bin/env python3
"""
QUICK START - Szybki test projektu
Uruchom ten plik aby sprawdzić czy wszystko działa poprawnie.
"""

print("=" * 80)
print("QUICK START - Test środowiska projektu")
print("=" * 80)

# Test 1: Python version
import sys
print(f"\n[OK] Python {sys.version.split()[0]}")

# Test 2: Import bibliotek
missing_packages = []

try:
    import pandas as pd
    print(f"[OK] pandas {pd.__version__}")
except ImportError:
    print("[BRAK] pandas")
    missing_packages.append("pandas")

try:
    import numpy as np
    print(f"[OK] numpy {np.__version__}")
except ImportError:
    print("[BRAK] numpy")
    missing_packages.append("numpy")

try:
    import sklearn
    print(f"[OK] scikit-learn {sklearn.__version__}")
except ImportError:
    print("[BRAK] scikit-learn")
    missing_packages.append("scikit-learn")

try:
    import lightgbm as lgb
    print(f"[OK] lightgbm {lgb.__version__}")
except ImportError:
    print("[BRAK] lightgbm")
    missing_packages.append("lightgbm")

try:
    import torch
    print(f"[OK] torch {torch.__version__}")
except ImportError:
    print("[BRAK] torch")
    missing_packages.append("torch")

try:
    import matplotlib
    print(f"[OK] matplotlib {matplotlib.__version__}")
except ImportError:
    print("[BRAK] matplotlib (opcjonalne)")

try:
    import seaborn
    print(f"[OK] seaborn {seaborn.__version__}")
except ImportError:
    print("[BRAK] seaborn (opcjonalne)")

# Test 3: Sprawdzenie danych
import os
print("\n" + "-" * 80)
print("Sprawdzanie struktury projektu:")

data_file = 'data/european_flights.csv'
if os.path.exists(data_file):
    print(f"[OK] Plik danych znaleziony: {data_file}")
    # Sprawdź rozmiar
    size_mb = os.path.getsize(data_file) / (1024 * 1024)
    print(f"  Rozmiar: {size_mb:.2f} MB")
else:
    print(f"[BRAK] Plik danych: {data_file}")
    print("  Umieść plik european_flights.csv w folderze data/")

# Test 4: Folder models
if os.path.exists('models'):
    print("[OK] Folder models/ istnieje")
else:
    print("[INFO] Folder models/ zostanie utworzony automatycznie")

# Test 5: main.py
if os.path.exists('main.py'):
    print("[OK] Plik main.py znaleziony")
else:
    print("[BRAK] Plik main.py!")

# Podsumowanie
print("\n" + "=" * 80)
if missing_packages:
    print("UWAGA: Brakujące pakiety:")
    for pkg in missing_packages:
        print(f"   - {pkg}")
    print("\nZainstaluj brakujące pakiety:")
    print(f"   pip install {' '.join(missing_packages)}")
    print("\nlub wszystkie naraz:")
    print("   pip install -r requirements.txt")
else:
    print("WSZYSTKO GOTOWE!")
    print("\nMożesz uruchomić projekt:")
    print("   python main.py")

print("=" * 80)
