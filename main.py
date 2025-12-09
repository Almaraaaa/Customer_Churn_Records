import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score

FIG_DIR = os.path.join('data', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

data = pd.read_csv('Customer-Churn-Records.csv')
print("Размер данных:", data.shape)

id_col = 'CustomerId' if 'CustomerId' in data.columns else None
ids = data[id_col] if id_col else data.index

for c in ['RowNumber', 'CustomerId', 'Surname']:
    if c in data.columns:
        data = data.drop(columns=c)

if 'Gender' in data.columns:
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

X = pd.get_dummies(data.drop(columns=['Exited']), drop_first=True)
X = X.fillna(X.median(numeric_only=True))
y = data['Exited'].astype(int)

print("Форма признаков:", X.shape)

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, ids, test_size=0.2, random_state=42, stratify=y
)

model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, verbose=0, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, os.path.join('data', 'model_catboost.joblib'))

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

results = X_test.copy().reset_index(drop=True)
results['CustomerId'] = ids_test.reset_index(drop=True)
results['Exited_true'] = y_test.reset_index(drop=True)
results['Exited_pred'] = y_pred
results['Exited_proba'] = y_proba
results.to_csv(os.path.join('data', 'churn_predictions_export.csv'), index=False)

importances = model.get_feature_importance()
feat = pd.Series(importances, index=X_train.columns).sort_values(ascending=False).head(15)
plt.figure(figsize=(8, max(4, len(feat)*0.35)))
sns.barplot(x=feat.values, y=feat.index)
plt.title('Топ признаков')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'feature_importances.png'), dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc_score(y_test, y_proba):.3f}')
plt.plot([0,1],[0,1],'--', color='grey')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.title('ROC Curve')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'roc_curve.png'), dpi=150)
plt.close()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()

