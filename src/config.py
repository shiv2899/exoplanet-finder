# Exoplanet Finder - config (Shallue & Vanderburg 2018, Astronet style)
# Paper: 0.988 AUC on Kepler Q1-Q17 DR24 TCE
DATA_CSV = "data/kepler_koi.csv"  # download, see data/README.md
GLOBAL_LEN = 2001
LOCAL_LEN = 201
# KOI columns we use
FEATURES_TABULAR = ["koi_period","koi_duration","koi_depth","koi_prad","koi_teq","koi_insol","koi_steff","koi_slogg","koi_srad"]
LABEL_COL = "koi_pdisposition"  # CANDIDATE vs FALSE POSITIVE
# Star split: same star (kepid) must never be in train AND test - no leakage
TEST_FRAC = 0.15
VAL_FRAC = 0.15
SEED = 42
BATCH = 64
EPOCHS = 30
LR = 1e-3
