"""Ön işleme. Fit edilmemiş döner; bir Pipeline içinde yalnız eğitim verisine fit edilir (sızıntı yok)."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),  # medyan aykırı km_driven'dan etkilenmez
        ("scale", StandardScaler()),                   # MLP ve Ridge ölçek farkına duyarlı
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # 20'den az görülen markalar tek "infrequent" sütununda toplanır; testte ilk kez görülen marka da oraya düşer.
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])
