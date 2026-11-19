import pytest

from cardekho.classification import (build_clf_searches, clf_results_table, evaluate_clf,
                                     get_xy_clf, mlp_experiment_table)
from cardekho.config import CLF_TARGET, REG_TARGET
from cardekho.data import clean, load_raw


@pytest.fixture(scope="module")
def fitted(sample_split):
    """Grid'ler bir kez fit edilir; aşağıdaki testler aynı sonuçları paylaşır."""
    searches = build_clf_searches()
    results = {k: evaluate_clf(s, *sample_split) for k, s in searches.items()}
    return searches, results


def test_target_not_in_clf_features():
    X, y = get_xy_clf(clean(load_raw()))
    assert REG_TARGET not in X.columns and CLF_TARGET not in X.columns
    assert set(y.unique()) == {0, 1}


def test_models_beat_baseline(fitted):
    _, res = fitted
    assert res["Random Forest"]["test_f1"] > res["Baseline"]["test_f1"]
    assert res["ANN (MLP)"]["test_f1"] > res["Baseline"]["test_f1"]


def test_results_table_columns(fitted):
    _, res = fitted
    tbl = clf_results_table(res)
    assert list(tbl.columns) == ["Model", "CV", "Eğitim", "Test", "Yorum"] and len(tbl) == 3
    assert (tbl["Yorum"] == "").all()


def test_mlp_experiment_table_has_every_config(fitted):
    searches, _ = fitted
    tbl = mlp_experiment_table(searches["ANN (MLP)"])
    assert len(tbl) == 6  # 3 mimari x 2 alpha
    assert {"Eğitim F1", "CV F1"} <= set(tbl.columns)
