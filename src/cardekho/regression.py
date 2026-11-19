"""Regresyon: selling_price (satış fiyatı, INR) tahmini."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from cardekho.config import CAT_FEATURES, FEATURES, NUM_FEATURES, REG_TARGET, SEED
from cardekho.preprocess import build_preprocessor


def get_xy_reg(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X'te transmission yok (sınıflandırma hedefi)."""
    return df[FEATURES], df[REG_TARGET]


def _pipe(model) -> Pipeline:
    return Pipeline([("pre", build_preprocessor(NUM_FEATURES, CAT_FEATURES)), ("model", model)])


def _log_target(model) -> TransformedTargetRegressor:
    """Fiyat sağa çarpık: model log1p(fiyat) üzerinde öğrenir, tahmin expm1 ile orijinal ölçeğe döner."""
    return TransformedTargetRegressor(regressor=_pipe(model), func=np.log1p, inverse_func=np.expm1)


def build_reg_searches() -> dict[str, GridSearchCV]:
    """CV skoru orijinal fiyat ölçeğinde RMSE'dir (TransformedTargetRegressor tahmini geri dönüştürür)."""
    models = {
        "Baseline": (_pipe(DummyRegressor(strategy="mean")), {}),
        "Ridge": (_log_target(Ridge()), {"regressor__model__alpha": [0.1, 1, 10, 100]}),
        "Random Forest": (
            _log_target(RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)),
            {"regressor__model__max_depth": [None, 10, 20], "regressor__model__min_samples_leaf": [1, 5]},
        ),
    }
    cv = KFold(5, shuffle=True, random_state=SEED)
    return {
        name: GridSearchCV(est, grid, cv=cv, scoring="neg_root_mean_squared_error", return_train_score=True)
        for name, (est, grid) in models.items()
    }


def _metrics(y, pred) -> dict:
    mse = mean_squared_error(y, pred)
    return {"mae": mean_absolute_error(y, pred), "mse": mse, "rmse": np.sqrt(mse),
            "r2": r2_score(y, pred), "mape": mean_absolute_percentage_error(y, pred)}


def evaluate_reg(search: GridSearchCV, X_tr, y_tr, X_te, y_te) -> dict:
    """Ayar seçimi yalnız eğitimde CV ile; test kümesi sadece en sonda bir kez tahmin edilir."""
    search.fit(X_tr, y_tr)
    return {
        "cv_rmse": -search.best_score_,
        "train": _metrics(y_tr, search.predict(X_tr)),
        "test": _metrics(y_te, search.predict(X_te)),
    }


def reg_results_table(results: dict[str, dict]) -> pd.DataFrame:
    """Ödevdeki tablo; ölçüt hücreleri "eğitim / test". Yorum sütunu öğrenci tarafından doldurulacak."""
    fmt = {"mae": "{:,.0f}", "mse": "{:.3e}", "rmse": "{:,.0f}", "r2": "{:.3f}", "mape": "{:.1%}"}
    names = {"mae": "MAE", "mse": "MSE", "rmse": "RMSE", "r2": "R²", "mape": "MAPE"}
    rows = []
    for model, r in results.items():
        row = {"Model": model, "CV": f"{r['cv_rmse']:,.0f}"}
        for key, f in fmt.items():
            row[names[key]] = f"{f.format(r['train'][key])} / {f.format(r['test'][key])}"
        row["Yorum"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def plot_actual_vs_pred(y_true, y_pred, path: Path) -> None:
    """Gerçek–tahmin grafiği. Fiyatlar birkaç büyüklük mertebesine yayıldığı için log-log eksen."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=8, alpha=0.4)
    lo, hi = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "r--", label="y = x (kusursuz tahmin)")
    ax.set(xscale="log", yscale="log", xlabel="Gerçek fiyat (INR)", ylabel="Tahmin (INR)",
           title="Gerçek – tahmin")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
