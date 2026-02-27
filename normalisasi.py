import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Membaca dataset yang sudah melalui proses data cleaning
df = pd.read_excel("Data_cleaning_fiks.xlsx")
print(df.columns)

# Kolom yang akan dinormalisasi
columns_to_normalize = [
    "Open", "High", "Low", "Close", "Volume",
    "MA_20", "MA_50", "RSI_14",
    "BB_Middle", "BB_Upper", "BB_Lower",
    "Return_%",
    "PER", "PBV", "ROE", "EPS",
]

# Membuat objek scaler
scaler = MinMaxScaler()

# Copy dataframe untuk menyimpan hasil normalisasi
df_normalized = df.copy()

# Proses normalisasi
df_normalized[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Simpan hasil normalisasi
df_normalized.to_excel("data_normalisasi.xlsx", index=False)

print("Normalisasi selesai! File tersimpan sebagai data_normalisasi.xlsx")
