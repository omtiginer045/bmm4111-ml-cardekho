"""Sınıflandırma: transmission (Manual / Automatic) tahmini."""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from cardekho.config import CAT_FEATURES, CLF_TARGET, FEATURES, NUM_FEATURES, SEED
from cardekho.preprocess import build_preprocessor


def get_xy_clf(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X'te selling_price yok (regresyon hedefi). Pozitif sınıf (1) Automatic."""
    return df[FEATURES], (df[CLF_TARGET] == "Automatic").astype(int)


def build_clf_searches() -> dict[str, GridSearchCV]:
    """Her model ön işlemeyle aynı Pipeline'da; CV her katmanda ön işlemeyi yalnız o katmanın eğitiminden öğrenir."""
    models = {
        "Baseline": (DummyClassifier(strategy="most_frequent"), {}),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
            {"model__max_depth": [None, 10, 20], "model__min_samples_leaf": [1, 5],
             "model__class_weight": [None, "balanced"]},
        ),
        "ANN (MLP)": (
            MLPClassifier(max_iter=500, early_stopping=True, random_state=SEED),
            {"model__hidden_layer_sizes": [(16,), (64,), (64, 32)], "model__alpha": [1e-4, 1e-2]},
        ),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return {
        name: GridSearchCV(
            Pipeline([("pre", build_preprocessor(NUM_FEATURES, CAT_FEATURES)), ("model", model)]),
            grid, cv=cv, scoring="f1_macro", return_train_score=True,
        )
        for name, (model, grid) in models.items()
    }


def evaluate_clf(search: GridSearchCV, X_tr, y_tr, X_te, y_te) -> dict:
    """Ayar seçimi yalnız eğitimde CV ile yapılır; test kümesi sadece en sonda bir kez tahmin edilir."""
    search.fit(X_tr, y_tr)
    pred_tr, pred_te = search.predict(X_tr), search.predict(X_te)
    return {
        "cv_f1": search.best_score_,
        "train_f1": f1_score(y_tr, pred_tr, average="macro"),
        "test_f1": f1_score(y_te, pred_te, average="macro"),
        "test_accuracy": accuracy_score(y_te, pred_te),
        "test_precision": precision_score(y_te, pred_te, average="macro", zero_division=0),
        "test_recall": recall_score(y_te, pred_te, average="macro"),
        "confusion_matrix": confusion_matrix(y_te, pred_te),
    }


def clf_results_table(results: dict[str, dict]) -> pd.DataFrame:
    """Ödevdeki tablo (ölçüt macro-F1). Yorum sütunu öğrenci tarafından el yazısıyla doldurulacak."""
    rows = [
        {"Model": name, "CV": round(r["cv_f1"], 3), "Eğitim": round(r["train_f1"], 3),
         "Test": round(r["test_f1"], 3), "Yorum": ""}
        for name, r in results.items()
    ]
    return pd.DataFrame(rows)


def mlp_experiment_table(search: GridSearchCV) -> pd.DataFrame:
    """Her MLP mimarisi/alpha için CV katmanlarındaki ortalama eğitim ve doğrulama F1'i."""
    cv = search.cv_results_
    return pd.DataFrame({
        "Mimari": [str(p["model__hidden_layer_sizes"]) for p in cv["params"]],
        "alpha": [p["model__alpha"] for p in cv["params"]],
        "Eğitim F1": cv["mean_train_score"].round(3),
        "CV F1": cv["mean_test_score"].round(3),
    })
