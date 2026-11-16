"""Plot Precision-Recall + ROC, save results/pr_roc.png"""
import pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
df=pd.read_csv("results/test_predictions.csv")
fig,ax=plt.subplots(1,2,figsize=(11,4))
for col,label in [("cnn_score","CNN"),("baseline_score","Baseline GB")]:
    p,r,_=precision_recall_curve(df.label,df[col]); ax[0].plot(r,p,label=f"{label} AP={auc(r,p):.3f}")
    f,t,_=roc_curve(df.label,df[col]); ax[1].plot(f,t,label=f"{label} AUC={auc(f,t):.3f}")
ax[0].set_title("Precision-Recall"); ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].set_title("ROC"); ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR"); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results/pr_roc.png",dpi=150); print("Saved -> results/pr_roc.png")
