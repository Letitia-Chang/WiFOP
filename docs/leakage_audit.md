# Leakage audit

Scope: read-through of `src/combine_data.py`, `src/clean_panel.py`, and the
training scripts (`src/train_xgboost_weighted.py`, `src/train_xgboost_smote.py`).
Caveat: as noted in [inventory.md](inventory.md), these two scripts operate on a
**July-2023-only local test panel** (filenames end in `_2023-07`); I did not find
a committed full 2020–2023 version. The issues below are structural (same author,
same logic would apply to the full run) but should be re-checked once the
full-scale script is located/rebuilt.

**Status (2026-11-23): all 4 issues fixed in `clean_panel.py` and the two
training scripts**, verified with synthetic data (the real NASA/NDVI/IRWIN
interim files aren't in this repo — see inventory.md — so this couldn't be
re-run against the actual July 2023 numbers). See each section below for
what changed.

## 1. Label/feature time alignment — the core "next-day" problem

**File:** [`src/combine_data.py:79-118`](../src/combine_data.py)

`fire_labels["fire_today"]` is built directly from IRWIN records on the same
`date`, and merged onto the panel on `(grid_id, date)` (line 115). Every weather,
NDVI, and fire-history feature computed afterward in `clean_panel.py` is also
keyed to that same `date`. So the model is trained to predict whether a fire
occurs **today**, using **today's** weather/NDVI — not whether a fire occurs
**tomorrow**, as the project brief states ("next-day wildfire ignition").

**Fix:** either (a) relabel the target as `fire_tomorrow` by shifting the fire
indicator back one day per `grid_id` (`groupby("grid_id")["fire_today"].shift(-1)`)
and drop the last date per cell, or (b) keep `fire_today` as the target but shift
all *features* forward one day (lag-1) so only information available at the end
of day *t* is used to predict day *t*. Either way, the "fire count in a cell over
the last 7/14 days" features described in the project brief should exclude the
current day.

**Done — option (a).** `clean_panel.py` now builds `fire_tomorrow` via
`groupby("grid_id")["fire_today"].shift(-1)` and drops the (NaN) last date per
cell. `train_xgboost_weighted.py` / `train_xgboost_smote.py` train on
`fire_tomorrow` instead of `fire_today`.

## 2. Fire-history features leak the same-day label into themselves

**File:** [`src/clean_panel.py:77-81`](../src/clean_panel.py)

```python
df["fires_last_7d"]  = df.groupby("grid_id")["fire_today"].transform(lambda s: s.rolling(7).sum())
df["fires_last_14d"] = df.groupby("grid_id")["fire_today"].transform(lambda s: s.rolling(14).sum())
```

`pandas.Series.rolling(window)` is **right-closed and inclusive of the current
row** by default. So `fires_last_7d` on day *t* includes `fire_today` on day *t*
itself: whenever `fire_today == 1`, `fires_last_7d >= 1` by construction. This
directly encodes the label into a "historical" feature and is used as a predictor
in every training script (`feature_cols` in `train_xgboost_weighted.py:31`,
`train_xgboost_smote.py`, etc.). This is likely the single largest source of the
reported scores — a feature that is partly a copy of the label will dominate
feature importance and inflate ROC-AUC/PR-AUC in a way that won't hold up
operationally (you don't know "today's" fire count before today happens).

**Fix:** shift before rolling, e.g.
`s.shift(1).rolling(7).sum()`, or equivalently `rolling(7, closed="left")` is not
supported pre-pandas 1.3 the same way — safest is
`df.groupby("grid_id")["fire_today"].transform(lambda s: s.shift(1).rolling(7).sum())`.
Re-run feature importance / SHAP after this fix — I'd expect the humidity/wind
oddity (fire days showing *higher* humidity, *lower* wind) to be partly explained
by this: if `fires_last_7d` is leaking the label, the model may be relying on it
instead of the weather features, and the raw EDA correlation between humidity and
same-day fire occurrence could simply reflect that CA fire season (Jun–Sep, per
the NASA POWER pulls) has its own humidity seasonality distinct from
daily fire-weather risk — worth checking with a lagged-weather plot once the
pipeline is fixed.

**Done.** `fires_last_7d`/`fires_last_14d` in `clean_panel.py` now use
`s.shift(1).rolling(window).sum()`. Same fix applied to `ndvi_7d_mean`, which
had the identical inclusivity issue (not just the separate future-leakage issue
in #3 below) — it's now `s.shift(1).rolling(7).mean()`. The SHAP re-run and
lagged-weather check are still open — they need the real data, not just the
synthetic data used to verify the code runs.

## 3. NDVI interpolation uses future data to fill the past

**File:** [`src/clean_panel.py:44-49`](../src/clean_panel.py)

```python
df["ndvi"] = df.groupby("grid_id")["ndvi"].transform(
    lambda x: x.interpolate("linear").bfill().ffill()
)
```

`interpolate("linear")` fills an interior gap at day *t* using surrounding known
values on *both* sides, including days after *t*. `.bfill()` explicitly fills
leading NaNs using the next *future* valid NDVI reading. Both leak future
satellite observations into a feature that is supposed to represent what was
knowable on day *t*. `ndvi_7d_mean` (`clean_panel.py:69`) is a rolling mean over
this already-leaked column, so it inherits the same issue, and (independently of
the leakage) also includes the current day's NDVI the same way `fires_last_7d`
does (point 2).

**Fix:** interpolate only using past and same-day values
(`interpolate("linear", limit_direction="forward")` applied from the first known
point) or forward-fill only (`.ffill()`), never `.bfill()`, for any feature meant
to be "known as of day t." If a grid cell's very first NDVI reading is missing,
drop those leading rows rather than back-filling from the future.

**Done.** `clean_panel.py` now does
`x.interpolate("linear", limit_direction="forward").ffill()` with no `.bfill()`.
Verified on a synthetic series: leading NaNs before a cell's first reading stay
NaN (and get dropped by the existing NASA-missing filter) instead of being
filled from a future value; interior gaps still interpolate correctly.

## 4. Validation split is random, not time- or space-based

**File:** [`src/train_xgboost_weighted.py:49-54`](../src/train_xgboost_weighted.py)
(same pattern in `train_xgboost_smote.py`)

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

This is a random row-level split of a single month (July 2023) of a spatio-
temporal panel. Rows from the same `grid_id` on adjacent days end up in both
train and validation, and the (leaky) `fires_last_7d`/`fires_last_14d` features
plus spatially autocorrelated weather/NDVI make train and validation rows
near-duplicates of each other. This isn't the train/test split described in the
project brief (train 2020–2022, test 2023) — it looks like a smaller local
debugging script — but the same random-split pattern should be checked in
whatever script produced the official 2023 holdout numbers; the brief's own
rework plan (item 3: spatial block CV) already anticipates this is also a
problem at the full-dataset scale, because fires cluster geographically
(Southern California, wildland-urban interface) and a date-only split still lets
geographically-adjacent cells leak across train/test.

**Done, partially.** Both training scripts now sort by `date` and split
chronologically (last 20% of dates → validation) instead of a random/stratified
row split. Since this panel only covers one month (2023-07), this is the closest
available fix to the brief's real plan (train 2020–2022 / test 2023) — that
proper year-based split still needs to happen once the full-scale
`combine_data.py`/`clean_panel.py` run exists. **Spatial block CV (adjacent
cells still leaking across the time split) is still open** — not addressed
here.

## Priority fix order

1. Fix the rolling-window inclusivity bug (#2) — it's a direct label leak and the
   cheapest fix (one `shift(1)`).
2. Decide and implement the label/feature alignment (#1) — this changes what
   "next-day" means throughout the pipeline, so it should land before anything
   else is re-tuned.
3. Fix NDVI interpolation (#3).
4. Replace the random split with a time-based (2020–2022 train / 2023 test) and
   eventually spatial block CV split (#4), per the project's existing rework plan.

After these fixes, expect PR-AUC to drop substantially from the original
0.085–0.087 — that's the point: the original numbers are inflated by leakage,
and the rewritten baselines (persistence, historical fire rate) need to be beaten
by a model trained without these shortcuts.
