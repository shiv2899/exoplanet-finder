"""Evaluate best CNN vs Baseline on UNSEEN STARS (test split)"""
import torch, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix, classification_report
from dataset import load_splits, make_views
from model import TwoViewCNN
from sklearn.ensemble import HistGradientBoostingClassifier
import config
from train import ExoDataset
from torch.utils.data import DataLoader

def main():
    tr,va,te=load_splits(config.DATA_CSV)
    print(f"Test on {len(te)} signals from {te.kepid.nunique()} UNSEEN stars")
    # Baseline retrain on train+val
    full=pd.concat([tr,va]); base_cols=["koi_period","koi_duration","koi_depth"]
    gb=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,random_state=42); gb.fit(full[base_cols],full["label"])
    p_base=gb.predict_proba(te[base_cols])[:,1]
    print(f"Baseline GB Test AUC={roc_auc_score(te.label,p_base):.4f} AP={average_precision_score(te.label,p_base):.4f}")
    # CNN
    model=TwoViewCNN(); model.load_state_dict(torch.load("checkpoints/cnn_best.pth",map_location="cpu")); model.eval()
    dl=DataLoader(ExoDataset(te),batch_size=128); preds=[]
    with torch.no_grad():
        for g,l,y,_ in dl: preds.extend(torch.sigmoid(model(g,l)).numpy())
    preds=np.array(preds)
    print(f"CNN Test AUC={roc_auc_score(te.label,preds):.4f} AP={average_precision_score(te.label,preds):.4f}")
    # Threshold at 0.5
    pred_label=(preds>0.5).astype(int)
    print(classification_report(te.label,pred_label,target_names=["FALSE POSITIVE","CANDIDATE"]))
    print("Confusion:", confusion_matrix(te.label,pred_label).tolist())
    pd.DataFrame({"kepid":te.kepid.values,"label":te.label.values,"cnn_score":preds,"baseline_score":p_base}).to_csv("results/test_predictions.csv",index=False)
    print("Saved -> results/test_predictions.csv")
if __name__=="__main__": main()
