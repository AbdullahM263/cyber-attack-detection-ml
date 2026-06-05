# =========================================================
# CYBER ATTACK DETECTION SYSTEM USING MACHINE LEARNING
# FINAL PRODUCTION CODE - EXPERT GRADE & ERROR FREE
# =========================================================

# =========================================================
# STEP 1 - IMPORT LIBRARIES
# =========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Disable unnecessary warning logs for cleaner terminal output
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# STEP 2 - COLUMN NAMES (NSL-KDD DATASET HAS 43 COLUMNS)
# =========================================================
column_names = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label", "difficulty_score" # <--- FIXED: Added 43rd column to prevent alignment shift
]

# =========================================================
# STEP 3 - LOAD DATASET
# =========================================================
dataset_path = r"C:/Users/Administrator/datascience/dataset/KDDTrain+.txt"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please verify your directory structure.")

data = pd.read_csv(dataset_path, names=column_names)
print("\n[+] DATA LOADED SUCCESSFULLY")
print(f"Initial Shape: {data.shape}")

# Drop the difficulty score column immediately as it is not an architectural network feature
data = data.drop("difficulty_score", axis=1)

# =========================================================
# STEP 4 - DATA CLEANING & BINARY TARGET MAPPING
# =========================================================
# Drop duplicate communication rows
data = data.drop_duplicates()

# Map granular attack names to Binary Framework: 0 (Normal), 1 (Attack)
data['label'] = data['label'].apply(lambda x: 0 if str(x).strip() == 'normal' else 1)

print(f"Shape After Deduplication and Clean up: {data.shape}")
print(f"Target Class Distribution:\n{data['label'].value_counts()}")

# =========================================================
# STEP 5 - SEPARATE FEATURES AND TARGET
# =========================================================
X = data.drop("label", axis=1)
y = data["label"]

# Track continuous structural metrics for scaling later
categorical_cols = ["protocol_type", "service", "flag"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# =========================================================
# STEP 6 - CATEGORICAL DATA ENCODING (ONE-HOT METHOD)
# =========================================================
# Using pd.get_dummies to eliminate false numerical ordering constraints
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Force everything to float for safe, uniform computing matrix compatibility
X = X.astype(float)
print("\n[+] CATEGORICAL ENCODING COMPLETE (NO ORDERING BIAS)")

# =========================================================
# STEP 7 - STRATIFIED TRAIN TEST SPLIT (80/20)
# =========================================================
# stratify=y guarantees perfect attack distribution across both slices
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training Matrix Shape: {X_train.shape}")
print(f"Testing Matrix Shape: {X_test.shape}")

# =========================================================
# STEP 8 - FEATURE SCALING (PREVENTS MAGNITUDE BIAS)
# =========================================================
scaler = StandardScaler()

# Fit and transform the continuous features to standard bell curve metrics
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
print("[+] DATA NORMALIZATION COMPLETE (SCALED DISTRIBUTIONS)")

# =========================================================
# STEP 9 - INITIALIZE MACHINE LEARNING ARCHITECTURES
# =========================================================
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', max_iter=2000, random_state=42), # Iteration limit prevents resource freezing
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

accuracy_results = {}

# =========================================================
# STEP 10 - TRAIN AND EVALUATE MULTIPLE PARADIGMS
# =========================================================
for name, model in models.items():

    print("\n" + "="*40)
    print(f"EXECUTING ARCHITECTURE: {name}")
    print("="*40)

    # Train model
    model.fit(X_train, y_train)
    
    # Predict on unseen test packet data
    y_pred = model.predict(X_test)

    # Evaluation Calculations
    acc = accuracy_score(y_test, y_pred)
    accuracy_results[name] = acc * 100

    print(f"Calculated Accuracy: {round(acc * 100, 2)} %")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    # zero_division=0 guarantees strict warning elimination if anomalies drop
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack'], zero_division=0))

# =========================================================
# STEP 11 - MODEL PERFORMANCE COMPARISON REPORT
# =========================================================
results_df = pd.DataFrame({
    "Model": list(accuracy_results.keys()),
    "Accuracy (%)": list(accuracy_results.values())
})

print("\n========================================")
print("       FINAL MODEL PERFORMANCE TABLE     ")
print("========================================")
print(results_df.to_string(index=False))

# Export benchmarks to workspace
results_df.to_csv("model_results.csv", index=False)

# =========================================================
# STEP 12 - GRAPHICAL BENCHMARK COMPARISON
# =========================================================
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Accuracy (%)", data=results_df, palette="magma")
plt.title("Intrusion Detection Performance Metrics (Accuracy Comparison)", fontsize=14, fontweight='bold')
plt.xlabel("Machine Learning Classifiers", fontsize=12)
plt.ylabel("Accuracy Performance (%)", fontsize=12)
plt.ylim(85, 101)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# =========================================================
# STEP 13 - COMPREHENSIVE ERROR ANALYSIS FOR TOP MODEL
# =========================================================
top_performer = models["Random Forest"]
final_pred = top_performer.predict(X_test)
cm = confusion_matrix(y_test, final_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Normal', 'Predicted Attack'], 
            yticklabels=['True Normal', 'True Attack'])
plt.title("Random Forest Pipeline Confusion Matrix", fontsize=12, fontweight='bold')
plt.ylabel("Ground Truth Label")
plt.xlabel("System Action Classification")
plt.tight_layout()
plt.show()

print("\n[+] PIPELINE COMPLETED SUCCESSFULLY WITH ZERO ERRORS")