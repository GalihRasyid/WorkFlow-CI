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

TOKEN_ASLI = "d1f669853cea910190197feb84d64f7cb5691026" 

os.environ["MLFLOW_TRACKING_USERNAME"] = "GalihRasyid"
os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN_ASLI
os.environ["DAGSHUB_USER_TOKEN"] = TOKEN_ASLI

try:
    add_app_token(TOKEN_ASLI)
except Exception:
    pass

dagshub.init(repo_owner='GalihRasyid', repo_name='submission_diabetes_GalihRasyid', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/GalihRasyid/submission_diabetes_GalihRasyid.mlflow")

# --- 2. LOAD DATA (FIX PATH) ---
# Kita buat logika pencarian file yang lebih cerdas
csv_filename = 'diabetes_clean.csv'

# Kemungkinan lokasi file:
# 1. Di folder yang sama (folder MLProject/diabetes_preprocessing)
# 2. Di folder tetangga (Eksperimen_SML.../preprocessing/diabetes_preprocessing)
possible_paths = [
    'diabetes_preprocessing/diabetes_clean.csv',  # Jika dicopy manual
    '../diabetes_preprocessing/diabetes_clean.csv', # Backup 1
    '../../Eksperimen_SML_GalihRasyid/preprocessing/diabetes_preprocessing/diabetes_clean.csv' # <--- INI LOKASI ASLINYA DI GITHUB
]

df = None
found_path = ""

for path in possible_paths:
    if os.path.exists(path):
        print(f"✅ Dataset ditemukan di: {path}")
        df = pd.read_csv(path)
        found_path = path
        break

if df is None:
    # Jika masih gagal, kita coba print isi folder saat ini untuk debugging
    print("❌ Error: Dataset tidak ditemukan dimanapun!")
    print(f"Posisi saat ini: {os.getcwd()}")
    print("Isi folder saat ini:", os.listdir())
    print("Mencoba naik satu level:", os.listdir('..'))
    raise FileNotFoundError("Gagal load diabetes_clean.csv")

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- 3. TRAINING & LOGGING (FIX KRITERIA 2) ---
mlflow.set_experiment("Diabetes_Fix_Artifacts")

with mlflow.start_run(run_name="Run_Fixed_Model"):
    # Matikan autolog biar kita manual saja (lebih pasti)
    mlflow.sklearn.autolog(disable=True)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"Accuracy: {acc}")
    
    # Log Metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("n_estimators", 100)
    
    # --- INI SOLUSINYA: SIMPAN MODEL SECARA EKSPLISIT ---
    # Ini akan membuat folder 'model' di DagsHub Artifacts
    mlflow.sklearn.log_model(model, "model") 
    
    # Simpan file lokal juga (untuk backup)
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")

    print("✅ Model berhasil disimpan ke MLflow Artifacts.")