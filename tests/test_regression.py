import pytest

from cardekho.config import CLF_TARGET, REG_TARGET
from cardekho.data import clean, load_raw
from cardekho.regression import (build_reg_searches, evaluate_reg, get_xy_reg,
                                 plot_actual_vs_pred, reg_results_table)


@pytest.fixture(scope="module")
def results(sample_split_reg):
    """Grid'ler bir kez fit edilir; aşağıdaki testler aynı sonuçları paylaşır."""
    return {k: evaluate_reg(s, *sample_split_reg) for k, s in build_reg_searches().items()}


def test_transmission_not_in_reg_features():
    X, _ = get_xy_reg(clean(load_raw()))
    assert CLF_TARGET not in X.columns and REG_TARGET not in X.columns


def test_baseline_r2_near_zero_and_models_better(results):
    res = results
    assert abs(res["Baseline"]["test"]["r2"]) < 0.05
    assert res["Random Forest"]["test"]["rmse"] < res["Baseline"]["test"]["rmse"]
    assert res["Ridge"]["test"]["r2"] > 0.5


def test_results_table_has_train_test_cells(results):
    tbl = reg_results_table(results)
    assert list(tbl.columns) == ["Model", "CV", "MAE", "MSE", "RMSE", "R²", "MAPE", "Yorum"]
    assert len(tbl) == 3 and (tbl["Yorum"] == "").all()
    assert tbl[["MAE", "MSE", "RMSE", "R²", "MAPE"]].map(lambda c: " / " in c).all().all()


def test_plot_actual_vs_pred_writes_file(tmp_path):
    path = tmp_path / "gercek_tahmin.png"
    plot_actual_vs_pred([100000, 500000, 900000], [120000, 450000, 1000000], path)
    assert path.exists() and path.stat().st_size > 0
