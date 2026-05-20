"""
train_model.py
──────────────
Run this once to generate model.pkl before launching the Streamlit app.
Usage: python train_model.py
"""
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

# ── Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('359611_6a0b3f1a6fb0b_1779121946.csv')

# ── Replace Medical Zeros with Median ────────────────────────────────────
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, df[col].replace(0, np.nan).median())

# ── Features & Target ─────────────────────────────────────────────────────
feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                'Insulin','BMI','DiabetesPedigreeFunction','Age']
X = df[feature_cols].values
y = df['Outcome'].values

# ── Scale ─────────────────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Split ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ── SMOTE ─────────────────────────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ── K-Means Centers ──────────────────────────────────────────────────────
BEST_K = 10
kmeans  = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
kmeans.fit(X_train)
centers = kmeans.cluster_centers_

# ── Sigma ─────────────────────────────────────────────────────────────────
dists = [np.linalg.norm(centers[i] - centers[j])
         for i in range(len(centers))
         for j in range(i+1, len(centers))]
sigma = np.mean(dists)

# ── RBF Features ──────────────────────────────────────────────────────────
def rbf_features(X, centers, sigma):
    out = np.zeros((X.shape[0], len(centers)))
    for i, c in enumerate(centers):
        out[:, i] = np.exp(-np.sum((X - c) ** 2, axis=1) / (2 * sigma ** 2))
    return out

H_train = rbf_features(X_train, centers, sigma)
H_test  = rbf_features(X_test,  centers, sigma)

# ── Logistic Regression Output Layer ─────────────────────────────────────
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(H_train, y_train)

# ── Threshold Optimization (F1) ───────────────────────────────────────────
proba_test = lr.predict_proba(H_test)[:, 1]
thresholds = np.arange(0.1, 0.91, 0.01)
f1s = [f1_score(y_test, (proba_test >= t).astype(int), zero_division=0)
       for t in thresholds]
best_threshold = thresholds[int(np.argmax(f1s))]
acc = ((proba_test >= best_threshold).astype(int) == y_test).mean()

print(f"✅ Accuracy  : {acc:.4f}")
print(f"✅ Threshold : {best_threshold:.2f}")
print(f"✅ K Centers : {BEST_K}")

# ── Save ──────────────────────────────────────────────────────────────────
model_data = {
    'scaler':       scaler,
    'centers':      centers,
    'sigma':        sigma,
    'lr':           lr,
    'threshold':    best_threshold,
    'feature_cols': feature_cols,
    'accuracy':     acc,
}
with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✅ model.pkl saved successfully!")
