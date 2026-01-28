import pandas as pd

# 1. Wczytanie danych
df = pd.read_excel("Wine_data_prep.xlsx")

# Sprawdzenie braków
print("Braki przed czyszczeniem:")
print(df.isnull().sum())
print(df.head())

# 2. Czyszczenie nazw kolumn
df.columns = df.columns.str.strip()

# 3. Konwersja kolumn do typu liczbowego, jeśli możliwe
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # błędne wartości zamienią się na NaN

# 4. Wypełnienie braków średnią tylko dla kolumn liczbowych
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

# Sprawdzenie braków po czyszczeniu
print("\nBraki po czyszczeniu:")
print(df.isnull().sum())

# 5. Sprawdzenie kolumn tekstowych i czy zawierają liczby
for col in df.select_dtypes(include='object'):
    print(col, df[col].str.isnumeric().sum())

# 6. Zapis do CSV
df.to_csv("wine_data.csv", index=False)

print("\nPlik wine_data.csv zapisany poprawnie!")
