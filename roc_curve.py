import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

np.random.seed(7)

n = 1000
y_true = np.random.binomial(1, 0.5, n)

y_score = np.where(
    y_true == 1,
    np.random.normal(0.65, 0.14, n),
    np.random.normal(0.35, 0.14, n)
)

y_score = np.clip(y_score, 0, 1)

fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

print("ROC-AUC:", round(roc_auc, 2))

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
