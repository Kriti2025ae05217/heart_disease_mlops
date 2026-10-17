"""Train, tune, cross-validate and package heart-disease classifiers.

    python -m src.heart.train                 # with MLflow (local ./mlruns)
    python -m src.heart.train --no-mlflow     # skip MLflow
"""
import argparse
import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)

from .data import FEATURES, ROOT, load_clean, split_xy, validate_schema
from .evaluate import (CV_SCORING, classification_metrics, save_confusion,
                       save_roc)
from .features import build_pipeline
from .tracking import Tracker

SEED = 42
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"


def candidate_models(seed: int = SEED) -> dict:
    """name -> (estimator, param grid). Grid keys are prefixed with the pipeline step."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=seed),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {"model__n_estimators": [100, 300],
             "model__max_depth": [3, 5, None],
             "model__min_samples_leaf": [1, 3]},
        ),
    }


def run_training(data_path=None, use_mlflow=True, tracking_uri=None,
                 seed=SEED, test_size=0.2, outer_folds=5, inner_folds=3) -> dict:
    df = load_clean(data_path) if data_path else load_clean()
    validate_schema(df)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)

    MODEL_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(enabled=use_mlflow, tracking_uri=tracking_uri)
    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
    inner = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)

    results, fitted = {}, {}
    with tracker.run("training-session"):
        tracker.params({"seed": seed, "test_size": test_size, "n_rows": len(df),
                        "outer_folds": outer_folds, "inner_folds": inner_folds,
                        "sklearn": sklearn.__version__,
                        "python": platform.python_version()})
        for name, (est, grid) in candidate_models(seed).items():
            with tracker.run(name, nested=True):
                search = GridSearchCV(build_pipeline(est), grid, cv=inner,
                                      scoring="roc_auc", n_jobs=-1)
                # Nested CV: honest generalisation estimate incl. hyper-parameter tuning
                cv = cross_validate(search, X_train, y_train, cv=outer,
                                    scoring=CV_SCORING, n_jobs=1)
                cv_metrics = {f"cv_{m}": cv[f"test_{m}"].mean() for m in CV_SCORING}
                cv_metrics.update({f"cv_{m}_std": cv[f"test_{m}"].std() for m in CV_SCORING})

                search.fit(X_train, y_train)
                best = search.best_estimator_
                proba = best.predict_proba(X_test)[:, 1]
                test_metrics = {f"test_{k}": v for k, v in classification_metrics(
                    y_test, best.predict(X_test), proba).items()}

                tracker.params({"model": name, **search.best_params_})
                tracker.metrics({**cv_metrics, **test_metrics,
                                 "best_inner_cv_roc_auc": search.best_score_})
                roc = save_roc(best, X_test, y_test, FIG_DIR / f"roc_{name}.png",
                               f"ROC - {name}")
                cm = save_confusion(best, X_test, y_test, FIG_DIR / f"cm_{name}.png",
                                    f"Confusion matrix - {name}")
                tracker.artifact(roc)
                tracker.artifact(cm)
                tracker.log_model(best, X_train, name="model")

                results[name] = {"best_params": search.best_params_,
                                 **{k: float(v) for k, v in cv_metrics.items()},
                                 **{k: float(v) for k, v in test_metrics.items()}}
                fitted[name] = best

        best_name = max(results, key=lambda n: results[n]["cv_roc_auc"])
        tracker.tag("selected_model", best_name)
        tracker.params({"selected_model": best_name})

    final = fitted[best_name]
    joblib.dump(final, MODEL_DIR / "model.joblib")
    meta = {
        "selected_model": best_name,
        "features": FEATURES,
        "metrics": results[best_name],
        "all_results": results,
        "sklearn_version": sklearn.__version__,
        "seed": seed,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    pd.DataFrame(results).T.to_csv(ROOT / "reports" / "model_comparison.csv")
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=None)
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    meta = run_training(a.data, not a.no_mlflow, a.tracking_uri, a.seed)
    print(json.dumps({"selected_model": meta["selected_model"],
                      "metrics": meta["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
