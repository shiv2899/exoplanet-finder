"""Train Two-view CNN + Baseline Gradient Boosting, save checkpoints + metrics"""
import argparse, numpy as np, torch, torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from dataset import load_splits, make_views
from model import TwoViewCNN
import config
from torch.utils.data import Dataset, DataLoader

class ExoDataset(Dataset):
    def __init__(self, df):
        self.df=df
        self.views=[make_views(r) for _,r in df.iterrows()]
        self.y=df["label"].values.astype(np.float32)
        self.tab=df[config.FEATURES_TABULAR].values.astype(np.float32)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        g,l=self.views[i]
        return torch.from_numpy(g), torch.from_numpy(l), torch.tensor(self.y[i]), torch.from_numpy(self.tab[i])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--epochs",type=int,default=30); ap.add_argument("--csv",default=config.DATA_CSV)
    a=ap.parse_args()
    tr,va,te=load_splits(a.csv)
    print(f"Train {len(tr)} Val {len(va)} Test {len(te)} | Train positives {tr.label.mean():.3f} Test positives {te.label.mean():.3f}")
    # --- Baseline: only period, duration, depth ---
    base_cols=["koi_period","koi_duration","koi_depth"]
    gb=HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42)
    gb.fit(tr[base_cols], tr["label"])
    for name,df in [("Val",va),("Test",te)]:
        p=gb.predict_proba(df[base_cols])[:,1]
        print(f"Baseline GB ({name}) AUC={roc_auc_score(df.label,p):.4f} AP={average_precision_score(df.label,p):.4f}")
    # --- CNN ---
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=TwoViewCNN().to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    loss_fn=nn.BCEWithLogitsLoss()
    tr_dl=DataLoader(ExoDataset(tr),batch_size=64,shuffle=True); va_dl=DataLoader(ExoDataset(va),batch_size=128)
    best=0
    for ep in range(a.epochs):
        model.train(); tot=0
        for g,l,y,_ in tr_dl:
            g,l,y=g.to(device),l.to(device),y.to(device)
            opt.zero_grad(); loss=loss_fn(model(g,l),y); loss.backward(); opt.step(); tot+=loss.item()
        model.eval(); preds=[]; ys=[]
        with torch.no_grad():
            for g,l,y,_ in va_dl:
                preds.extend(torch.sigmoid(model(g.to(device),l.to(device))).cpu().numpy()); ys.extend(y.numpy())
        auc=roc_auc_score(ys,preds); ap=average_precision_score(ys,preds)
        print(f"Epoch {ep+1}/{a.epochs} loss={tot/len(tr_dl):.4f} Val AUC={auc:.4f} AP={ap:.4f}")
        if auc>best: best=auc; torch.save(model.state_dict(),"checkpoints/cnn_best.pth")
    print(f"Best Val AUC {best:.4f} saved to checkpoints/cnn_best.pth")

if __name__=="__main__": main()
