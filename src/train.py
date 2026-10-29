"""Steps 3-4: model training, evaluation and MLflow experiment tracking.

Trains 3 classifiers (Good_Investment) and 3 regressors (Future_Price_5Y), logs every run to
MLflow, and saves the best model of each task to models/ for the Streamlit app.

Run:   python src/train.py
View:  mlflow ui --backend-store-uri sqlite:///mlflow.db   (then open http://127.0.0.1:5000)
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             mean_absolute_error, mean_absolute_percentage_error,
                             precision_score, r2_score, recall_score, roc_auc_score,
                             root_mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from features import MODEL_CATEGORICAL, MODEL_NUMERIC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "housing_clean.csv"
MODELS = ROOT / "models"
REPORT = ROOT / "reports" / "model_results.md"
SEED = 42

mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
EXPERIMENT = "real_estate_investment_advisor"
if mlflow.get_experiment_by_name(EXPERIMENT) is None:
    mlflow.create_experiment(EXPERIMENT, artifact_location=(ROOT / "mlartifacts").as_uri())
mlflow.set_experiment(EXPERIMENT)


def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), MODEL_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), MODEL_CATEGORICAL),
    ])


def feature_importance(pipe):
    names = pipe.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        imp = np.abs(np.ravel(model.coef_))
    s = pd.Series(imp, index=names)
    # roll one-hot columns back up to their original feature
    def base(n):
        for c in MODEL_CATEGORICAL:
            if n.startswith(c + "_"):
                return c
        return n
    s = s.groupby(base).sum()
    return (s / s.sum()).sort_values(ascending=False)


def plot_importance(imp, title, path):
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values, color="#3a7ca5")
    ax.set(title=title, xlabel="Relative importance")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    df = pd.read_csv(DATA)
    X = df[MODEL_NUMERIC + MODEL_CATEGORICAL]
    y_clf, y_reg = df["Good_Investment"], df["Future_Price_5Y"]
    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=SEED, stratify=y_clf)
    print(f"Train {len(X_tr):,} / test {len(X_te):,}")

    tmp = ROOT / "reports" / "figures" / "models"
    tmp.mkdir(parents=True, exist_ok=True)
    results = {"classification": [], "regression": []}

    # ------------------------------------------------------ classification
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100, max_depth=14, min_samples_leaf=10, max_samples=0.3,
            n_jobs=-1, random_state=SEED),
        "XGBClassifier": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, random_state=SEED, eval_metric="logloss"),
    }
    best_clf, best_f1 = None, -1
    for name, model in classifiers.items():
        pipe = Pipeline([("prep", preprocessor()), ("model", model)])
        with mlflow.start_run(run_name=f"clf_{name}"):
            t = time.time(); pipe.fit(X_tr, yc_tr); train_s = time.time() - t
            pred = pipe.predict(X_te); proba = pipe.predict_proba(X_te)[:, 1]
            m = {"accuracy": accuracy_score(yc_te, pred), "precision": precision_score(yc_te, pred),
                 "recall": recall_score(yc_te, pred), "f1": f1_score(yc_te, pred),
                 "roc_auc": roc_auc_score(yc_te, proba), "train_seconds": train_s}
            mlflow.set_tag("task", "classification")
            mlflow.log_params({"model": name, **{k: v for k, v in model.get_params().items()
                                                 if isinstance(v, (int, float, str)) and v is not None}})
            mlflow.log_metrics(m)
            fig, ax = plt.subplots(figsize=(4.5, 4))
            ConfusionMatrixDisplay.from_predictions(yc_te, pred, ax=ax, cmap="Blues",
                                                    display_labels=["Not good", "Good"])
            ax.set_title(name); fig.tight_layout()
            cm_path = tmp / f"cm_{name}.png"; fig.savefig(cm_path, dpi=110); plt.close(fig)
            mlflow.log_artifact(str(cm_path))
            imp = feature_importance(pipe)
            imp_path = tmp / f"importance_{name}.png"
            plot_importance(imp, f"{name} — feature importance", imp_path)
            mlflow.log_artifact(str(imp_path))
            info = mlflow.sklearn.log_model(pipe, name="model", input_example=X_te.head(3),
                                            serialization_format="cloudpickle")
            print(f"  {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))
            results["classification"].append({"model": name, **m})
            if m["f1"] > best_f1:
                best_f1, best_clf, best_clf_name, best_clf_imp = m["f1"], pipe, name, imp
                best_clf_uri = info.model_uri

    # ------------------------------------------------------ regression
    regressors = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100, max_depth=14, min_samples_leaf=10, max_samples=0.3,
            n_jobs=-1, random_state=SEED),
        "XGBRegressor": XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.08, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, random_state=SEED),
    }
    best_reg, best_rmse = None, np.inf
    for name, model in regressors.items():
        pipe = Pipeline([("prep", preprocessor()), ("model", model)])
        with mlflow.start_run(run_name=f"reg_{name}"):
            t = time.time(); pipe.fit(X_tr, yr_tr); train_s = time.time() - t
            pred = pipe.predict(X_te)
            m = {"rmse": root_mean_squared_error(yr_te, pred), "mae": mean_absolute_error(yr_te, pred),
                 "r2": r2_score(yr_te, pred), "mape": mean_absolute_percentage_error(yr_te, pred),
                 "train_seconds": train_s}
            mlflow.set_tag("task", "regression")
            mlflow.log_params({"model": name, **{k: v for k, v in model.get_params().items()
                                                 if isinstance(v, (int, float, str)) and v is not None}})
            mlflow.log_metrics(m)
            fig, ax = plt.subplots(figsize=(5, 5))
            idx = np.random.default_rng(SEED).choice(len(pred), 5000, replace=False)
            ax.scatter(yr_te.values[idx], pred[idx], s=4, alpha=0.4)
            lim = [yr_te.min(), yr_te.max()]; ax.plot(lim, lim, "r--", lw=1)
            ax.set(title=f"{name}: actual vs predicted", xlabel="Actual (lakhs)", ylabel="Predicted (lakhs)")
            fig.tight_layout(); ap = tmp / f"actual_vs_pred_{name}.png"; fig.savefig(ap, dpi=110); plt.close(fig)
            mlflow.log_artifact(str(ap))
            imp = feature_importance(pipe)
            imp_path = tmp / f"importance_{name}.png"
            plot_importance(imp, f"{name} — feature importance", imp_path)
            mlflow.log_artifact(str(imp_path))
            info = mlflow.sklearn.log_model(pipe, name="model", input_example=X_te.head(3),
                                            serialization_format="cloudpickle")
            print(f"  {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))
            results["regression"].append({"model": name, **m})
            if m["rmse"] < best_rmse:
                best_rmse, best_reg, best_reg_name, best_reg_imp = m["rmse"], pipe, name, imp
                best_reg_uri = info.model_uri

    # ------------------------------------------------------ register + save best models
    mlflow.register_model(best_clf_uri, "good_investment_classifier")
    mlflow.register_model(best_reg_uri, "future_price_regressor")
    joblib.dump(best_clf, MODELS / "classifier.joblib", compress=3)
    joblib.dump(best_reg, MODELS / "regressor.joblib", compress=3)
    (MODELS / "model_meta.json").write_text(json.dumps({
        "best_classifier": best_clf_name, "best_regressor": best_reg_name,
        "results": results,
        "clf_importance": best_clf_imp.round(4).to_dict(),
        "reg_importance": best_reg_imp.round(4).to_dict(),
    }, indent=1))

    lines = ["# Model results\n", f"Test set: {len(X_te):,} properties (20% stratified hold-out).\n",
             "## Classification — Good_Investment\n",
             "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time (s) |",
             "|---|---|---|---|---|---|---|"]
    for r in results["classification"]:
        lines.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
                     f"{r['f1']:.4f} | {r['roc_auc']:.4f} | {r['train_seconds']:.0f} |")
    lines += [f"\nSelected: **{best_clf_name}** (highest F1).\n",
              "## Regression — Future_Price_5Y (lakhs)\n",
              "| Model | RMSE | MAE | R² | MAPE | Train time (s) |", "|---|---|---|---|---|---|"]
    for r in results["regression"]:
        lines.append(f"| {r['model']} | {r['rmse']:.3f} | {r['mae']:.3f} | {r['r2']:.5f} | "
                     f"{r['mape']:.2%} | {r['train_seconds']:.0f} |")
    lines.append(f"\nSelected: **{best_reg_name}** (lowest RMSE).\n")
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
