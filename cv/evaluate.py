"""
evaluate.py — Comprehensive accuracy & threshold evaluation for the Gaze Tracking project.

Covers:
  1. ML Model accuracy (Logistic Regression & Random Forest)
  2. Per-class precision / recall / F1
  3. Confusion matrix heatmap
  4. Blink detection threshold analysis (EAR ratio sweep)
  5. Gaze direction threshold sweep (DX / DY dead-zone sensitivity)
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg')   # headless backend - no display required
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE = os.path.dirname(__file__)
DATASET_PATH  = os.path.join(BASE, "eye_movement_dataset.csv")
RF_MODEL_PATH = os.path.join(BASE, "gaze_blink_rf_model.pkl")
SCALER_PATH   = os.path.join(BASE, "feature_scaler.pkl")
OUT_DIR       = os.path.join(BASE, "eval_output")
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_MAP  = {0: "LEFT", 1: "RIGHT", 2: "UP", 3: "DOWN", 4: "CENTER"}
CLASS_NAMES = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
print("=" * 60)
print("  GAZE TRACKING - ACCURACY & THRESHOLD EVALUATION")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
print(f"\n[*] Dataset: {len(df)} samples")
print(f"   Label distribution:\n{df['LABEL'].value_counts().sort_index().to_string()}\n")

X = df[["EAR_L", "EAR_R", "EAR_AVG", "DX", "DY"]].values
y = df["LABEL"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 2. Train & Evaluate Both Models
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
}

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    acc    = accuracy_score(y_test, y_pred)

    # Cross-validation accuracy (5-fold)
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='accuracy')

    print(f"--- {name} -------------------------------------------")
    print(f"  Test Accuracy      : {acc*100:.2f}%")
    print(f"  5-Fold CV Accuracy : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"\n{classification_report(y_test, y_pred, target_names=CLASS_NAMES)}")

# ─────────────────────────────────────────────
# 3. Confusion Matrix — Random Forest
# ─────────────────────────────────────────────
rf = models["Random Forest"]
y_pred_rf = rf.predict(X_test_s)
cm = confusion_matrix(y_test, y_pred_rf)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Random Forest - Confusion Matrix", fontsize=14, fontweight='bold')

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
axes[0].set_title("Raw Counts")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
axes[1].set_title("Normalised (Recall per class)")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"[OK] Saved confusion matrix -> {cm_path}")

# ─────────────────────────────────────────────
# 4. Blink Detection Threshold Analysis
#    Sweep EAR ratio from 0.60 to 0.95
# ─────────────────────────────────────────────
print("\n--- Blink Threshold (EAR Ratio) Sweep -----------------")

# For rows labeled as blink (DOWN has lowest EAR in this dataset),
# compare against non-blink rows using avg EAR.
# We compute TP/FP rates across different ratio thresholds.

ear_col = df["EAR_AVG"].values
# Use CENTER as baseline (eyes open) — label 4
center_mask = (df["LABEL"] == 4).values
blink_mask  = (df["LABEL"] == 3).values  # DOWN tends to have lower EAR

base_ear = ear_col[center_mask].mean()
print(f"  Baseline EAR (CENTER class mean) : {base_ear:.4f}")

ratios = np.linspace(0.60, 0.98, 40)
tp_rates, fp_rates, accuracies_blink = [], [], []

for ratio in ratios:
    thresh = base_ear * ratio
    pred_blink = ear_col < thresh
    tp = np.sum(pred_blink & blink_mask)
    fn = np.sum(~pred_blink & blink_mask)
    fp = np.sum(pred_blink & center_mask)
    tn = np.sum(~pred_blink & center_mask)
    tpr = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    tp_rates.append(tpr); fp_rates.append(fpr)
    # Blink detection accuracy on blink+center rows only
    relevant = blink_mask | center_mask
    acc_b = (np.sum((pred_blink == True)  & blink_mask  & relevant) +
             np.sum((pred_blink == False) & center_mask & relevant)) / relevant.sum()
    accuracies_blink.append(acc_b)

best_idx = np.argmax(accuracies_blink)
best_ratio = ratios[best_idx]
print(f"  Best EAR Ratio Threshold         : {best_ratio:.2f}  (accuracy = {accuracies_blink[best_idx]*100:.1f}%)")
print(f"  Current code uses ratio          : 0.85  (BLINK_THRESH_RATIO)")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(ratios, [a*100 for a in accuracies_blink], 'b-o', ms=3, label='Blink Accuracy (%)')
ax.axvline(0.85, color='red', linestyle='--', label='Current threshold (0.85)')
ax.axvline(best_ratio, color='green', linestyle='--', label=f'Optimal ({best_ratio:.2f})')
ax.set_xlabel("EAR Ratio Threshold"); ax.set_ylabel("Accuracy (%)")
ax.set_title("Blink Detection Accuracy vs EAR Ratio Threshold")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
blink_path = os.path.join(OUT_DIR, "blink_threshold_sweep.png")
plt.savefig(blink_path, dpi=150)
plt.close()
print(f"[OK] Saved blink threshold sweep -> {blink_path}")

# ─────────────────────────────────────────────
# 5. Gaze DX / DY Threshold Sensitivity
# ─────────────────────────────────────────────
print("\n─── Gaze Direction DX/DY Threshold Sweep ──────────────")

dx = df["DX"].values
dy = df["DY"].values
labels = df["LABEL"].values

dx_thresholds = np.linspace(0.01, 0.15, 30)
dy_thresholds = np.linspace(0.01, 0.18, 30)

# Lateral (LEFT=0 / RIGHT=1) accuracy vs DX threshold
lr_accs = []
for th in dx_thresholds:
    lr_mask = (labels == 0) | (labels == 1)
    if lr_mask.sum() == 0: continue
    pred_right = dx[lr_mask] > th
    pred_left  = dx[lr_mask] < -th
    correct = ((pred_right) & (labels[lr_mask] == 1)) | ((pred_left) & (labels[lr_mask] == 0))
    lr_accs.append(correct.sum() / lr_mask.sum())

# Vertical (UP=2 / DOWN=3) accuracy vs DY threshold
ud_accs = []
for th in dy_thresholds:
    ud_mask = (labels == 2) | (labels == 3)
    if ud_mask.sum() == 0: continue
    pred_down = dy[ud_mask] > th
    pred_up   = dy[ud_mask] < -th
    correct = ((pred_down) & (labels[ud_mask] == 3)) | ((pred_up) & (labels[ud_mask] == 2))
    ud_accs.append(correct.sum() / ud_mask.sum())

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(dx_thresholds[:len(lr_accs)], [a*100 for a in lr_accs], 'b-o', ms=3)
axes[0].axvline(0.04, color='red', linestyle='--', label='Current H_THRESH (0.04)')
axes[0].set_xlabel("DX Threshold"); axes[0].set_ylabel("L/R Accuracy (%)")
axes[0].set_title("LEFT/RIGHT Detection vs DX Threshold")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(dy_thresholds[:len(ud_accs)], [a*100 for a in ud_accs], 'g-o', ms=3)
axes[1].axvline(0.06, color='red', linestyle='--', label='Current UP_THRESH (0.06)')
axes[1].set_xlabel("DY Threshold"); axes[1].set_ylabel("U/D Accuracy (%)")
axes[1].set_title("UP/DOWN Detection vs DY Threshold")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
gaze_path = os.path.join(OUT_DIR, "gaze_threshold_sweep.png")
plt.savefig(gaze_path, dpi=150)
plt.close()
print(f"[OK] Saved gaze threshold sweep -> {gaze_path}")

# ─────────────────────────────────────────────
# 6. Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  ML Model Accuracy (test set) : {accuracy_score(y_test, y_pred_rf)*100:.2f}%")
print(f"  ML Model CV Accuracy (5-fold): {cross_val_score(rf, scaler.transform(X), y, cv=5).mean()*100:.2f}%")
print(f"  Optimal Blink EAR Ratio      : {best_ratio:.2f}  (current: 0.85)")
print(f"\n  Output plots saved to: {OUT_DIR}")
print("=" * 60)
