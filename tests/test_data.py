from cardekho.data import load_raw


def test_raw_shape_and_columns():
    df = load_raw()
    assert df.shape == (8128, 13)
    assert list(df.columns) == ["name", "year", "selling_price", "km_driven", "fuel",
        "seller_type", "transmission", "owner", "mileage", "engine", "max_power", "torque", "seats"]
