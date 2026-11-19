import numpy as np

from cardekho.config import CAT_FEATURES, CLF_TARGET, FEATURES, NUM_FEATURES
from cardekho.data import clean, load_raw
from cardekho.preprocess import build_preprocessor
from cardekho.split import make_split


def test_split_is_deterministic_and_stratified():
    df = clean(load_raw())
    tr1, te1 = make_split(df); tr2, te2 = make_split(df)
    assert tr1.index.equals(tr2.index) and te1.index.equals(te2.index)
    assert set(tr1.index).isdisjoint(te1.index)
    p_tr = (tr1[CLF_TARGET] == "Automatic").mean(); p_te = (te1[CLF_TARGET] == "Automatic").mean()
    assert abs(p_tr - p_te) < 0.01


def test_preprocessor_unseen_brand_and_nan(raw_small):
    df = clean(raw_small)
    pre = build_preprocessor(NUM_FEATURES, CAT_FEATURES).fit(df[FEATURES])
    new = df[FEATURES].head(1).copy(); new["brand"] = "Bilinmeyen"; new["max_power"] = np.nan
    out = pre.transform(new)
    assert not np.isnan(out).any()


def test_scaler_learns_only_from_train():
    df = clean(load_raw()); tr, te = make_split(df)
    pre = build_preprocessor(NUM_FEATURES, []).fit(tr[NUM_FEATURES])
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    assert np.allclose(scaler.mean_[0], tr["year"].mean())
