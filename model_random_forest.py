import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================================================
# 1. LOAD DATA
# ======================================================
df = pd.read_excel("data_normalisasi.xlsx")
df["Date"] = pd.to_datetime(df["Date"])

print("Jumlah data awal:", df.shape)
print("Kolom:", df.columns.tolist())

# ======================================================
# 2. DEFINISI FITUR
# ======================================================
features = [
    "Open", "High", "Low", "Volume",
    "MA_20", "MA_50", "RSI_14",
    "BB_Middle", "BB_Upper", "BB_Lower",
    "Return_%", "ROE", "EPS", "PER", "PBV"
]

target = "Close"

# ======================================================
# 3. PISAHKAN DATA TRAIN & DATA HARI INI
# ======================================================

# Data historis (Close ADA)
df_train = df[df["Close"].notna()].copy()

# Data hari ini (Close KOSONG)
df_today = df[df["Close"].isna()].copy()

print("Data train:", df_train.shape)
print("Data hari ini:", df_today.shape)

# ======================================================
# 4. SIAPKAN X & y UNTUK TRAINING
# ======================================================
X = df_train[features]
y = df_train[target]

# Pastikan tidak ada NaN
data_train = pd.concat([X, y], axis=1).dropna()
X = data_train[features]
y = data_train[target]

# ======================================================
# 5. SPLIT DATA (70:30)
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    shuffle=True
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# ======================================================
# 6. MODEL RANDOM FOREST
# ======================================================
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

# ======================================================
# 7. TRAINING
# ======================================================
model.fit(X_train, y_train)

# ======================================================
# 8. EVALUASI MODEL
# ======================================================
y_test_pred = model.predict(X_test)

print("\n=== HASIL EVALUASI MODEL ===")
print(f"MAE : {mean_absolute_error(y_test, y_test_pred):.4f}")
print(f"MSE : {mean_squared_error(y_test, y_test_pred):.4f}")
print(f"R²  : {r2_score(y_test, y_test_pred):.4f}")

# ======================================================
# 9. PREDIKSI HARGA PENUTUPAN HARI INI
# ======================================================
if len(df_today) > 0:

    X_today = df_today[features]

    # WAJIB cek NaN
    print("\nNaN pada fitur hari ini:")
    print(X_today.isna().sum())

    if X_today.isna().sum().sum() == 0:
        df_today["Predicted_Close"] = model.predict(X_today)
    else:
        raise ValueError("❌ Fitur hari ini masih mengandung NaN")

else:
    print("⚠️ Tidak ada data hari ini untuk diprediksi")

# ======================================================
# 10. SIMPAN HASIL KE EXCEL
# ======================================================

# Hasil training & testing
df_train_result = df_train.loc[X_train.index, ["Date", "Ticker"]].copy()
df_train_result["Actual_Close"] = y_train.values
df_train_result["Predicted_Close"] = model.predict(X_train)
df_train_result["Dataset"] = "Train"

df_test_result = df_train.loc[X_test.index, ["Date", "Ticker"]].copy()
df_test_result["Actual_Close"] = y_test.values
df_test_result["Predicted_Close"] = y_test_pred
df_test_result["Dataset"] = "Test"

# Hasil prediksi hari ini
if len(df_today) > 0:
    df_today_result = df_today[["Date", "Ticker", "Predicted_Close"]].copy()
    df_today_result["Dataset"] = "Today"
    final_result = pd.concat([df_train_result, df_test_result, df_today_result])
else:
    final_result = pd.concat([df_train_result, df_test_result])

final_result.to_excel("nyoba_ke-5.xlsx", index=False)

print("\n✅ File disimpan: nyoba_idx30.xlsx")