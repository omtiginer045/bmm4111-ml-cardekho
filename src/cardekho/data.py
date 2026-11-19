"""Ham verinin yüklenmesi, birim parse'ı ve temizlenmesi."""
from pathlib import Path

import numpy as np
import pandas as pd

from cardekho.config import CLF_TARGET, FEATURES, RAW_PATH, REG_TARGET

# owner'ın doğal bir sırası var: test sürüşü aracı < 1. sahip < ... < 4+ sahip
OWNER_ORDER = {
    "Test Drive Car": 0,
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Ham CarDekho CSV'sini olduğu gibi okur (temizlik yok)."""
    return pd.read_csv(path)


def parse_number(s: pd.Series) -> pd.Series:
    """'23.4 kmpl', '1248 CC' gibi metinlerden baştaki sayıyı alır.

    Sayı yoksa (boş, ' bhp') veya değer 0 ise NaN döner: 0 km/l ya da 0 bhp fiziksel olarak geçersiz.
    """
    out = s.astype(str).str.extract(r"([\d.]+)")[0].astype(float)
    return out.replace(0, np.nan)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Plandaki veri kararlarını uygular. Eksik değerler burada doldurulmaz (imputer pipeline içinde)."""
    # Aynı satır hem train'de hem test'te olmasın diye tekrarlar bölmeden önce silinir.
    df = raw.drop_duplicates()
    # CNG/LPG'de mileage km/kg cinsinden; kmpl ile karşılaştırılamaz ve çok az satır var.
    df = df[~df["fuel"].isin(["CNG", "LPG"])].copy()
    # ~2000 benzersiz model adı yerine marka (ilk kelime) kullanılır.
    df["brand"] = df["name"].str.split().str[0]
    for col in ["mileage", "engine", "max_power"]:
        df[col] = parse_number(df[col])
    df["owner"] = df["owner"].map(OWNER_ORDER)
    # name ve torque atılınca yalnız yazımı farklı ilanlar birebir aynı olur; tekrar silinir.
    df = df[FEATURES + [CLF_TARGET, REG_TARGET]].drop_duplicates()
    return df.reset_index(drop=True)
