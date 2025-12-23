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

# --- 1. SETUP AUTHENTICATION ---
# Catatan: Idealnya token disimpan di GitHub Secrets, tapi untuk sekarang hardcode tidak apa-apa agar jalan.
TOKEN_ASLI = os.environ.get("DAGSHUB_TOKEN")

if not TOKEN_ASLI:
    raise ValueError("Token DagsHub tidak ditemukan! Pastikan sudah diset di Secrets.")

os.environ["MLFLOW_TRACKING_USERNAME"] = "GalihRasyid" # Ganti username Anda
os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN_ASLI
os.environ["DAGSHUB_USER_TOKEN"] = TOKEN_ASLI

try:
    add_app_token(TOKEN_ASLI)
except Exception:
    pass

# Init DagsHub
dagshub.init(repo_owner='GalihRasyid', repo_name='submission_diabetes_GalihRasyid', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/GalihRasyid/submission_diabetes_GalihRasyid.mlflow")

# --- 2. LOAD DATA (SIMPLIFIED & FIXED) ---
print("📂 Memulai proses loading data...")

# Karena di GitHub Actions file csv sudah ada di sebelah script ini,
# kita panggil langsung namanya.
csv_filename = 'diabetes_clean.csv'

if os.path.exists(csv_filename):
    print(f"✅ Dataset ditemukan: {csv_filename}")
    df = pd.read_csv(csv_filename)
else:
    # Debugging jika file tidak ketemu
    print(f"❌ Error: File '{csv_filename}' tidak ditemukan di folder saat ini.")
    print(f"Posisi folder kerja (CWD): {os.getcwd()}")
    print(f"Isi folder saat ini: {os.listdir()}")
    raise FileNotFoundError(f"Gagal load {csv_filename}. Pastikan file ada di folder yang sama dengan script.")

# --- 3. PREPARE DATA ---
# Pastikan nama kolom target sesuai dataset Anda ('Outcome' atau 'target')
# Kita gunakan 'Outcome' sesuai kode Anda sebelumnya.
if 'Outcome' in df.columns:
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
elif 'target' in df.columns: # Fallback jika namanya target
    print("⚠️ Kolom 'Outcome' tidak ada, menggunakan 'target'.")
    X = df.drop('target', axis=1)
    y = df['target']
else:
    raise ValueError("Kolom Target (Outcome/target) tidak ditemukan di CSV!")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. TRAINING & LOGGING ---
print("🚀 Memulai Training Model...")
mlflow.set_experiment("Diabetes_Fix_Artifacts")

with mlflow.start_run(run_name="Run_Fixed_Model_CI"):
    # Matikan autolog agar kita bisa log manual dengan rapi
    mlflow.sklearn.autolog(disable=True)

    # Train Model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"📊 Accuracy: {acc}")
    
    # Log Metrics & Params Manual
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # --- SIMPAN MODEL (ARTIFACTS) ---
    # 1. Log model langsung ke MLflow Registry
    mlflow.sklearn.log_model(model, "model") 
    
    # 2. Simpan file lokal .pkl (Penting untuk Artifact Upload di GitHub Actions)
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl") # Upload pkl juga ke DagsHub sebagai backup

    print("✅ Model berhasil disimpan (model.pkl & MLflow Artifacts).")