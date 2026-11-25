# Leakage audit

Scope: read-through of `src/combine_data.py`, `src/clean_panel.py`, and the
training scripts (`src/train_xgboost_weighted.py`, `src/train_xgboost_smote.py`).
Caveat: as noted in [inventory.md](inventory.md), these two scripts operate on a
**July-2023-only local test panel** (filenames end in `_2023-07`); I did not find
a committed full 2020–2023 version. The issues below are structural (same author,
same logic would apply to the full run) but should be re-checked once the
full-scale script is located/rebuilt.

**Status (2026-11-23): all 4 issues fixed, and the full pipeline has now
been run end-to-end on real data for the full 2020–2023 range** (not just
synthetic data, and not just the July-2023 test panel) — see
`docs/inventory.md` for how the real NASA POWER/IRWIN/NDVI/NLCD data was
obtained and `docs/results.md` for the resulting model metrics. See each
section below for what changed in the code.

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

**Done.** Now that the full 2020–2023 pipeline exists and has been run on real
data, both training scripts implement the brief's actual plan: test is the
full held-out 2023 fire season, train is 2020–2022 (a true year-based split,
not a within-month chronological cut). Within the 2020–2022 training pool,
the validation set used for threshold tuning is a **spatial block split**
(`clean_panel.py` tags every grid cell with a `block_id` from a coarse
~0.5°/~55km lat/lon bin; 20% of distinct blocks become validation), so
geographically adjacent cells no longer end up on both sides of train/val.
See `docs/results.md` for the resulting metrics (2023 test ROC-AUC 0.93
weighted / 0.90 SMOTE).

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

## 5. Fire-label date and type semantics (critique item #6, 2026-11-25)

Two separate questions about what the fire label (`date`, from WFIGS'
`FireDiscoveryDateTime`) and the incident set actually represent, raised as
an open item in the post-leakage-audit critique (see `docs/results.md`) and
checked here against the live WFIGS_Incident_Locations FeatureServer schema
and real 2020–2023 CA data.

### 5a. Is `FireDiscoveryDateTime` an ignition date? (Not fixable — a data
limitation, not a bug)

**No.** Per NIFC/IRWIN's own field semantics (the InFORM data dictionary's
definition, and corroborated by the field's name), `FireDiscoveryDateTime`
is *"the date and time a fire was reported as discovered or confirmed to
exist,"* not an ignition timestamp. **WFIGS has no ignition-date field at
all** — querying the live FeatureServer schema (97 fields) turns up
`FireDiscoveryDateTime`, `ContainmentDateTime`, `ControlDateTime`,
`FireOutDateTime`, `InitialResponseDateTime`, `CreatedOnDateTime_dt`,
`ModifiedOnDateTime_dt`, and several ICS-209-report timestamps, but nothing
representing "when the fire actually started." This is a known, structural
limitation of wildfire record-keeping in general (most fires, especially
remote/wildland ones, are noticed some time after actual ignition — by a
lookout, satellite detection, or a public report) and applies equally to
every other public wildfire dataset (FPA-FOD, FIRMS, etc.) — it is not
something introduced by this project or fixable from this data source.
This project's "next-day" framing should be read as "next day after a fire
would first be reported/confirmed," which is the honest framing of what
the label can support, not "next day after ignition."

Checked for a confound at the *database* level (is there at least a
reporting-lag issue between discovery and when WFIGS records it?): sampled
2,000 real 2023 CA incidents and compared `FireDiscoveryDateTime` against
`CreatedOnDateTime_dt` (when the system created the record). Median lag
0.008 days (~11 minutes), 90th percentile 0.046 days (~1 hour) — at the
daily granularity this whole pipeline operates on, discovery and database
recording are effectively same-day for the vast majority of records (one
115-day outlier, likely a backfilled correction, and 12 negative-lag
records, likely rounding/backdating — both negligible in count). So there's
no material *additional* lag beyond the discovery-vs-ignition gap already
discussed above.

### 5b. Is every record actually a wildfire? (Fixable — a real bug, fixed)

**No — found a real label-contamination bug.** The download script
(`scripts/download_irwin_wfigs.py`) filtered only on `POOState='US-CA'`
and the date range; it never filtered on `IncidentTypeCategory`. Re-querying
the live FeatureServer with the exact same filters the script already used
(same CA bounding box, same date range) and grouping by
`IncidentTypeCategory` on the 49,539 rows previously downloaded:

| IncidentTypeCategory | Count | % |
|---|---|---|
| WF (wildfire) | 48,303 | 97.5% |
| RX (prescribed/planned burn) | 1,214 | 2.45% |
| CX (complex — aggregation of multiple incidents) | 22 | 0.04% |

**2.45% of the "fire" labels this entire project has been trained and
evaluated against are prescribed burns** — planned, intentionally set fires
with fundamentally different timing dynamics (chosen for a safe weather
window, often pre-announced) than the wildfire ignitions this project is
actually trying to predict. CX is negligible in count and left out of this
analysis (a complex is itself an aggregation, not a distinct incident type
with different dynamics).

**Fix:** added `IncidentTypeCategory='WF'` to the download script's `WHERE`
clause. This is a genuine, if small, mislabeling bug — not caught by the
original leakage audit because the audit focused on time alignment, not on
whether the label's *content* (what counts as a "fire") was correct.

**Impact not yet re-run.** At 2.45% of positive labels in an already
extremely rare-event setting (0.16% base rate), this could move metrics by
a small but non-negligible amount — rare-event models are sensitive to
small changes in an already-tiny positive set, and prescribed burns likely
have distinguishable weather/timing signatures (a planned burn needs
specific humidity/wind conditions to be legally permitted) that may have
been easier for the model to rank highly, inflating apparent skill
slightly. Re-downloading labels, rebuilding the clean panel, and retraining
all five models is deliberately **not done yet** (the 2.45% contamination
rate makes a large shift in the headline conclusions unlikely, and a
pipeline rebuild + full retrain + re-evaluation is substantial compute) —
flagged here as a known, fixed-at-the-source bug whose downstream effect on
`docs/results.md`'s existing numbers is not yet measured.

## 6. Three more bugs found while preparing the WFIGS-fix rerun (2026-11-25)

Found while setting up the pipeline rerun that was supposed to just apply
the item-5 WFIGS fix — none were caught by the original audit or its
regression tests (the third, 6c, was found immediately while fixing 6b).
All three are now fixed in `src/panel_transforms.py` (one only partially —
see 6c), with new regression tests in `tests/test_panel_transforms.py`
confirmed to catch each bug (verified by temporarily reintroducing each
one and watching the corresponding test fail, same discipline as item
#7's reproducibility harness).

### 6a. The NDVI "forward-only" fix (#3) didn't actually hold for interior gaps

**The bug:** `interpolate_ndvi` used
`x.interpolate("linear", limit_direction="forward").ffill()`. The
`limit_direction` argument only controls which end of a run of NaNs gets
*extrapolated* (i.e. leading/trailing gaps with no known value on one
side) — it does **not** make interior interpolation forward-only. For a
gap strictly between two known readings (e.g. `[0.4, NaN, NaN, 1.0]`),
pandas' linear interpolation always uses **both** endpoints regardless of
`limit_direction`, producing a ramp (`0.6, 0.8`) that leaks the *future*
`1.0` reading into the earlier NaN days. The original leading-gap test
(`test_ndvi_interpolation_never_backfills_leading_gap`) happened to only
exercise the one case `limit_direction` genuinely does control, so it kept
passing while this was broken.

**Fix:** replaced `.interpolate("linear", ...).ffill()` with plain
`.ffill()` — carries the last known value forward only, never computes a
value using a later reading. This is a real, if small, behavior change:
previously-interpolated interior gaps now hold flat at the last known
value instead of ramping.

### 6b. Every rolling/shift operation crossed fire-season boundaries

**The bug:** the raw panel only has rows for the fire season (roughly
Jun-Sep) each year — there's no Oct-May data at all. So within one grid
cell's rows sorted by date, September of year N is immediately followed
by June of year N+1 in **row position**, despite being ~8 months apart in
**time**. Every transform in this file grouped by `grid_id` alone, and
`.shift()`/`.rolling()` operate on row position, not elapsed time — so:

- `fires_last_7d`/`14d`, `ndvi_7d_mean`, and the raw weather rolling
  features (`T2M_3d_mean`, etc.) at the start of a new season could pull
  in fire/weather/NDVI data from the tail end of the *previous* season, 8
  months earlier.
- **More seriously:** `fire_tomorrow` on the last day of every
  non-final year's season (e.g. Sep 30, 2020/2021/2022) was being set to
  the **following year's June 1st** fire status, instead of being dropped
  as undefined (there's no real "tomorrow" after a season ends). On the
  real panel this affected 78,585 rows (26,195 cells × 3 year-boundaries),
  102 of them mislabeled as positive.

**Fix:** every grouping in `panel_transforms.py` is now `(grid_id, year)`
via a shared `_season_groups()` helper, not `grid_id` alone. The
consequence for `build_fire_tomorrow_label`: the last day of *every*
season (not just the single last day of the whole multi-year series) is
now correctly dropped. The consequence for the rolling features: the
first few days of each season now have fewer valid prior values within
their own window (correctly — there genuinely isn't a window to look back
on at the start of a new season), which produces a few more `NaN`
weather-rolling values at each season's start that get dropped by the
downstream `dropna` in each training script (see item 3 below for why
`fires_last_7d`/`14d` specifically need a different fix, not just a drop).

### 6c. `fires_last_7d`/`14d` silently reported "no fires" during the gap it couldn't fix on its own (found 2026-11-25, same prep session)

**The bug, found immediately after fixing 6b:** `.rolling(window)` with
the pandas default `min_periods=window` returns `NaN` whenever fewer than
`window` values are available — which, combined with the season-boundary
fix above, is now true for the first `window-1` days of *every* season
(there genuinely aren't 7/14 same-season prior rows yet). The old code
then did `.fillna(0)`, silently turning "not enough history to know" into
"zero fires" — actively wrong, not just missing, whenever a real fire had
already happened earlier in that same (now-shorter) window. E.g. a fire 2
days into a season would be invisible to `fires_last_7d` on day 5, when it
should clearly still count.

**Fix (partial — see limitation below):** `.rolling(window, min_periods=1)`
instead of the default `min_periods=window` — sums whatever same-season
prior days ARE available (1 to `window-1` of them) rather than requiring a
full window before reporting anything. A known recent fire within the
current season is no longer hidden.

**Known remaining limitation, not fixed:** the raw panel only has rows for
the fire season itself (no Oct-May rows at all), so a fire in the real
gap month immediately before a season starts (e.g. late May) has no row
in this dataframe to be rolled over — `min_periods=1` only helps with
days that exist in this panel, not days that don't. The complete fix
would mean computing fire history from the raw incident dates (which DO
cover the full calendar year, unlike this grid×date panel) before
restricting to fire-season rows — not done; flagged in
`docs/results.md`'s "Still open" list rather than silently assumed fixed.

### Separately flagged, not fixed: UTC day boundary vs. local fire-weather day

`scripts/download_irwin_wfigs.py` computes `date` from
`FireDiscoveryDateTime` as a UTC timestamp's calendar day
(`pd.to_datetime(..., utc=True).dt.strftime("%Y-%m-%d")`). California is
UTC-7/UTC-8; a fire discovered after ~4-5pm local time gets dated the
*following* UTC day. NASA POWER's daily values are computed on a
different day convention (not confirmed to be local solar time vs. UTC
for this project's download). This means an evening fire's label date
could be off by one day relative to the weather features it's paired
with — a real possible issue, but confirming and fixing it requires
pinning down NASA POWER's exact day-boundary convention, which hasn't
been done. Flagged here rather than silently assumed correct; see
`docs/results.md`'s "Still open" list.

### Status

All three fixed (6c only partially — see its limitation note) in code and
covered by regression tests as of this writing. **Re-run through the full
pipeline 2026-11-25**, combined with the item-5 WFIGS incident-type fix in
one rerun (see `docs/results.md`'s "Final rerun with all leakage fixes
applied" section for the resulting numbers and caveats) — isolating each
fix's individual numerical contribution would have roughly doubled the
compute cost for limited portfolio value; each fix is independently
justified as a correctness fix regardless of its measured effect size.

Result: the headline finding (historical fire rate beats every trained
model; the genuine day-to-day skill on top is modest) is unchanged after
the combined fix. Old-vs-new models on common rows are statistically
indistinguishable in 7 of 9 paired bootstrap comparisons; the two that
exclude zero (CatBoost ROC-AUC, Random Forest PR-AUC) are both tiny and not
treated as material. See `docs/results.md` for the full numbers and the
caveats that still apply (bundled fixes, different row sets than earlier
numbers in this document, unlimited NDVI ffill, late-May fire-history
blind spot, unfixed UTC day-boundary issue, no training-randomness term in
any CI, and stale baseline/ablation/tuning/within-cell artifacts that still
reflect the old panel).
