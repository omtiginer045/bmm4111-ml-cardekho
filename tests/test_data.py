import pandas as pd

from cardekho.config import CLF_TARGET, FEATURES, REG_TARGET
from cardekho.data import clean, load_raw, parse_number


def test_raw_shape_and_columns():
    df = load_raw()
    assert df.shape == (8128, 13)
    assert list(df.columns) == ["name", "year", "selling_price", "km_driven", "fuel",
        "seller_type", "transmission", "owner", "mileage", "engine", "max_power", "torque", "seats"]


def test_parse_number_handles_units_blanks_and_zero():
    s = pd.Series(["23.4 kmpl", "1248 CC", "74 bhp", " bhp", "", None, "0", "0.0 kmpl"])
    out = parse_number(s)
    assert out.iloc[:3].tolist() == [23.4, 1248.0, 74.0]
    assert out.iloc[3:].isna().all()


def test_clean_rules(raw_small):
    df = clean(raw_small)
    assert "torque" not in df and "name" not in df and "brand" in df
    assert not df["fuel"].isin(["CNG", "LPG"]).any()
    assert df.duplicated().sum() == 0
    assert set(df["owner"].dropna()) <= {0, 1, 2, 3, 4}
    assert list(df.columns) == FEATURES + [CLF_TARGET, REG_TARGET]


def test_clean_real_data_meets_assignment():
    df = clean(load_raw())
    assert len(df) >= 1000 and len(FEATURES) >= 8
    assert df[CLF_TARGET].nunique() == 2
    # 8128 → 6926 (tam tekrar) → 6832 (CNG/LPG) → 6813 (ad/torque farkıyla tekrar)
    assert len(df) == 6813


def test_iqr_outliers_bounds_and_counts():
    from cardekho.data import iqr_outliers

    df = pd.DataFrame({"a": [1, 2, 3, 4, 100, None], "b": [10, 10, 10, 10, 10, 10]})
    tbl = iqr_outliers(df, ["a", "b"])
    assert list(tbl.columns) == ["sütun", "Q1", "Q3", "alt_sınır", "üst_sınır", "aykırı_sayısı", "aykırı_oranı"]
    a = tbl.set_index("sütun").loc["a"]
    # Q1=2, Q3=4, IQR=2 → sınırlar -1 ve 7; yalnız 100 aykırı, NaN sayılmaz
    assert (a["alt_sınır"], a["üst_sınır"], a["aykırı_sayısı"]) == (-1.0, 7.0, 1)
    assert tbl.set_index("sütun").loc["b", "aykırı_sayısı"] == 0
    assert len(df) == 6  # girdi değiştirilmez, satır silinmez
