# Data - Kepler KOI
1. Go to NASA Exoplanet Archive -> Kepler Objects of Interest (KOI) -> Download CSV
   Direct: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&format=csv
2. Save as `data/kepler_koi.csv` (keep header row starting with # or plain header)
3. Columns needed: kepid, koi_pdisposition, koi_period, koi_duration, koi_depth, koi_prad, koi_teq, koi_insol, koi_steff, koi_slogg, koi_srad

Note: Original paper uses Q1-Q17 DR24 TCE table (15,737 TCEs). KOI table is the public equivalent with same disposition labels.
