"""Ham verinin yüklenmesi."""
from pathlib import Path

import pandas as pd

from cardekho.config import RAW_PATH


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Ham CarDekho CSV'sini olduğu gibi okur (temizlik yok)."""
    return pd.read_csv(path)
