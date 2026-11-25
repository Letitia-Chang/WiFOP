# Runs the repo's tune_models.py unchanged, but stores trials in SQLite so a timeout doesn't lose them.
import sys, runpy, optuna
m = sys.argv[sys.argv.index("--model") + 1]
_orig = optuna.create_study
def create_study(**kw):
    kw.setdefault("storage", f"sqlite:///../reports/metrics/optuna_{m}.db")
    kw.setdefault("study_name", m); kw.setdefault("load_if_exists", False)
    return _orig(**kw)
optuna.create_study = create_study
sys.argv[0] = "tune_models.py"; sys.path.insert(0, ".")
runpy.run_path("tune_models.py", run_name="__main__")
