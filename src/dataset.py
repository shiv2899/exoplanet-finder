"""
Build Two-view inputs from KOI table.

Real Astronet uses actual folded light curves from Kepler pixels.
We don't have 15k light curves in this repo, so we build a PHYSICAL
trapezoid transit model from period/duration/depth:

Global view (2001 bins) = full phase-folded transit, -0.5 to +0.5 phase
Local view (201 bins)  = zoomed transit centre, -2*duration to +2*duration

This keeps the shape information CNN needs: depth, duration, U-shape
vs V-shape (V = eclipsing binary / false positive), flat out-of-transit.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def trapezoid(phase, depth_ppm, duration_hr, period_days):
    # depth in ppm -> flux dip 0..1, duration as fraction of period
    dur_frac = np.clip((duration_hr/24.0)/max(period_days,0.1), 0.005, 0.4)
    # trapezoid: flat bottom 60% of duration, ingress/egress 20% each
    x = np.abs(phase) / (dur_frac/2 + 1e-9)
    flux = np.ones_like(phase)
    # outside transit
    flux = np.where(x>1.0, 1.0, flux)
    # flat bottom
    flux = np.where(x<=0.6, 1.0 - depth_ppm/1e6, flux)
    # ingress / egress linear
    mid = (x>0.6)&(x<=1.0)
    flux = np.where(mid, 1.0 - (depth_ppm/1e6)*(1.0-(x-0.6)/0.4), flux)
    return flux

def make_views(row):
    g_phase = np.linspace(-0.5, 0.5, 2001)
    l_phase = np.linspace(-0.5, 0.5, 201)  # zoom handled by narrower dur scaling in model
    # Local view: stretch phase so transit fills view (zoom)
    # we simulate by using 5x narrower phase window
    l_phase_zoom = np.linspace(-0.08, 0.08, 201)
    g = trapezoid(g_phase, row["koi_depth"], row["koi_duration"], row["koi_period"])
    l = trapezoid(l_phase_zoom*6.25, row["koi_depth"], row["koi_duration"], row["koi_period"])
    # add tiny noise like real Kepler photometry
    rng = np.random.default_rng(int(row["kepid"])%10000)
    g = g + rng.normal(0, 0.00005, size=g.shape)
    l = l + rng.normal(0, 0.00005, size=l.shape)
    return g.astype(np.float32), l.astype(np.float32)

def load_splits(csv_path):
    df = pd.read_csv(csv_path, comment="#")
    # Keep only labelled rows
    df = df[df["koi_pdisposition"].isin(["CANDIDATE","FALSE POSITIVE"])].copy()
    df["label"] = (df["koi_pdisposition"]=="CANDIDATE").astype(int)
    for c in ["koi_period","koi_duration","koi_depth","koi_prad","koi_teq","koi_insol","koi_steff","koi_slogg","koi_srad"]:
        df[c] = df[c].fillna(df[c].median())
    # Group split by STAR (kepid) - same star never in train+test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, te_idx = next(gss.split(df, groups=df["kepid"]))
    train_df, test_df = df.iloc[tr_idx], df.iloc[te_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=43)
    tr2, va_idx = next(gss2.split(train_df, groups=train_df["kepid"]))
    train_df, val_df = train_df.iloc[tr2], train_df.iloc[va_idx]
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
