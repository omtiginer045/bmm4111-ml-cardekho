import numpy as np
import pandas as pd
import pytest

COLS = ["name", "year", "selling_price", "km_driven", "fuel", "seller_type", "transmission",
        "owner", "mileage", "engine", "max_power", "torque", "seats"]


@pytest.fixture
def raw_small():
    """Temizleme kurallarının her birini tetikleyen küçük sahte ham veri."""
    rows = [
        # normal satır ve birebir tekrarı (bölmeden önce silinmeli)
        ["Maruti Swift VDI", 2014, 450000, 145500, "Diesel", "Individual", "Manual", "First Owner",
         "23.4 kmpl", "1248 CC", "74 bhp", "190Nm@ 2000rpm", 5],
        ["Maruti Swift VDI", 2014, 450000, 145500, "Diesel", "Individual", "Manual", "First Owner",
         "23.4 kmpl", "1248 CC", "74 bhp", "190Nm@ 2000rpm", 5],
        # CNG ve LPG (mileage birimi km/kg) atılmalı
        ["Maruti Wagon R CNG", 2016, 300000, 60000, "CNG", "Individual", "Manual", "First Owner",
         "33.54 km/kg", "998 CC", "67 bhp", "90Nm@ 3500rpm", 5],
        ["Hyundai Santro LPG", 2010, 150000, 90000, "LPG", "Individual", "Manual", "Second Owner",
         "18.9 kmpl", "1086 CC", "62 bhp", "96Nm@ 3000rpm", 5],
        # max_power boş birim ve "0" → NaN
        ["Hyundai i20 Asta", 2012, 350000, 80000, "Petrol", "Dealer", "Manual", "Second Owner",
         "18.5 kmpl", "1197 CC", " bhp", "115Nm@ 4000rpm", 5],
        ["Tata Indica V2", 2008, 100000, 120000, "Diesel", "Individual", "Manual", "Third Owner",
         "17.0 kmpl", "1405 CC", "0", "135Nm@ 2500rpm", 5],
        # mileage "0.0 kmpl" → NaN
        ["Honda City ZX", 2007, 200000, 110000, "Petrol", "Individual", "Manual", "First Owner",
         "0.0 kmpl", "1497 CC", "78 bhp", "12.7@ 2,700(kgm@ rpm)", 5],
        # çoğu sayısal alan eksik, nadir owner değeri
        ["Mahindra XUV500 W8", 2019, 1500000, 10, "Diesel", "Dealer", "Manual", "Test Drive Car",
         np.nan, np.nan, np.nan, np.nan, np.nan],
        # yalnız ad/torque yazımı farklı çift: name ve torque atılınca tekrar olur
        ["Toyota Fortuner 3.0", 2015, 2000000, 70000, "Diesel", "Individual", "Automatic",
         "Fourth & Above Owner", "11.5 kmpl", "2982 CC", "168.5 bhp", "343Nm@ 1400rpm", 7],
        ["Toyota Fortuner 3.0 4x2", 2015, 2000000, 70000, "Diesel", "Individual", "Automatic",
         "Fourth & Above Owner", "11.5 kmpl", "2982 CC", "168.5 bhp", "343Nm@ 1400-3400rpm", 7],
    ]
    return pd.DataFrame(rows, columns=COLS)


@pytest.fixture(scope="session")
def sample_train_test():
    """Gerçek bölmeden SEED ile stratified 1500 train / 400 test örneklemi (testler hızlı olsun diye)."""
    from sklearn.model_selection import train_test_split

    from cardekho.config import CLF_TARGET, SEED
    from cardekho.data import clean, load_raw
    from cardekho.split import make_split

    tr, te = make_split(clean(load_raw()))
    tr = train_test_split(tr, train_size=1500, stratify=tr[CLF_TARGET], random_state=SEED)[0]
    te = train_test_split(te, train_size=400, stratify=te[CLF_TARGET], random_state=SEED)[0]
    return tr, te


@pytest.fixture(scope="session")
def sample_split(sample_train_test):
    from cardekho.classification import get_xy_clf

    tr, te = sample_train_test
    return (*get_xy_clf(tr), *get_xy_clf(te))


@pytest.fixture(scope="session")
def sample_train_df(sample_train_test):
    return sample_train_test[0]


@pytest.fixture(scope="session")
def sample_split_reg(sample_train_test):
    from cardekho.regression import get_xy_reg

    tr, te = sample_train_test
    return (*get_xy_reg(tr), *get_xy_reg(te))
