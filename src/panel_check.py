"""Read-only summary of the raw and clean panels. Run from WiFOP/src."""
import pandas as pd
R="../data/processed/california_5km_daily_panel_2020-2023.csv"; C="../data/processed/california_5km_training_panel_2020-2023_clean.csv"
r=pd.read_csv(R,usecols=["grid_id","date","fire_today","PRECTOTCORR","ndvi","pct_11"])
print("RAW rows",len(r),"cells",r.grid_id.nunique(),"dates",r.date.nunique(),"fire_today",int(r.fire_today.sum()),f"rate {r.fire_today.mean():.6%}","missing",r[["PRECTOTCORR","ndvi","pct_11"]].isna().mean().round(4).to_dict(),"dupes",int(r.duplicated(["grid_id","date"]).sum())); del r
import sys
sys.path.insert(0,".")
from features import BASE_FEATURES
d=pd.read_csv(C,parse_dates=["date"]); y=d.date.dt.year
print("CLEAN rows",len(d),"cols",d.shape[1],"cells",d.grid_id.nunique(),"fire_today",int(d.fire_today.sum()),f"{d.fire_today.mean():.6%}","fire_tomorrow",int(d.fire_tomorrow.sum()),f"{d.fire_tomorrow.mean():.6%}")
ok=d[BASE_FEATURES].notna().all(axis=1); print("model rows",int(ok.sum()),"by year",ok.groupby(y).sum().to_dict(),"fires",d.fire_tomorrow[ok].groupby(y[ok]).sum().to_dict())
print("Sep30 rows",int(((d.date.dt.month==9)&(d.date.dt.day==30)).sum()),"by year rows",y.value_counts().sort_index().to_dict()); print("NaN share",d[BASE_FEATURES].isna().mean().loc[lambda s:s>0].round(4).to_dict())
