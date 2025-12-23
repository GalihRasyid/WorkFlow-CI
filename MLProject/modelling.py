import os
import joblib
import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dagshub.auth import add_app_token

# --- 1. SETUP AUTHENTICATION (SMART CHECK) ---

# Cek apakah kita sedang di GitHub Actions / MLflow Run (Environment Variable sudah ada?)
# Jika MLFLOW_TRACKING_URI sudah ada (dari main.yml), kita TIDAK PERLU dagshub.init
if "MLFLOW_TRACKING_URI" not in os.environ:
    print("⚠️ Running Locally: Menggunakan dagshub.init...")
    
    # Setup manual untuk lokal
    TOKEN_ASLI = os.environ.get("DAGSHUB_TOKEN")
    if not TOKEN_ASLI:
         # Fallback token hardcode HANYA untuk test lokal jika env var kosong
         # (Jangan lupa hapus/ganti ini jika ingin benar-benar bersih)
         TOKEN_ASLI = "d1f669853cea910190197feb84d64f7cb5691026"

    os.environ["MLFLOW_TRACKING_USERNAME"] = "GalihRasyid"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN_ASLI
    
    try:
        add_app_token(TOKEN_ASLI)
    except Exception:
        pass

    dagshub.init(repo_owner='GalihRasyid', repo_name='submission_diabetes_GalihRasyid', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/GalihRasyid/submission_diabetes_GalihRasyid.mlflow")
else:
    print("✅ Running in CI/CD: MLflow Tracking URI terdeteksi dari Environment.")

# --- 2. LOAD DATA ---
print("📂 Memulai proses loading data...")
csv_filename = 'diabetes_clean.csv'

if os.path.exists(csv_filename):
    print(f"✅ Dataset ditemukan: {csv_filename}")
    df = pd.read_csv(csv_filename)
else:
    print(f"❌ Error: File '{csv_filename}' tidak ditemukan di folder saat ini.")
    print(f"Posisi folder kerja (CWD): {os.getcwd()}")
    print(f"Isi folder saat ini: {os.listdir()}")
    raise FileNotFoundError(f"Gagal load {csv_filename}.")

# --- 3. PREPARE DATA ---
if 'Outcome' in df.columns:
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
elif 'target' in df.columns:
    X = df.drop('target', axis=1)
    y = df['target']
else:
    raise ValueError("Kolom Target tidak ditemukan!")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. TRAINING & LOGGING ---
print("🚀 Memulai Training Model...")

# PENTING: Jangan set_experiment di sini jika pakai 'mlflow run --experiment-name'
# Tapi untuk keamanan (agar lokal tetap jalan), kita set nama yang sama.
mlflow.set_experiment("Diabetes_Fix_Artifacts")

with mlflow.start_run(run_name="Run_Fixed_Model_CI"):
    mlflow.sklearn.autolog(disable=True)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"📊 Accuracy: {acc}")
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Simpan Model
    mlflow.sklearn.log_model(model, "model") 
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")

    print("✅ Model berhasil disimpan.")