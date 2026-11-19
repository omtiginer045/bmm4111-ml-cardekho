"""Tüm görevlerin paylaştığı tek eğitim/test bölmesi."""
import pandas as pd
from sklearn.model_selection import train_test_split

from cardekho.config import CLF_TARGET, SEED, TEST_SIZE


def make_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """%80/%20 böler. Automatic sınıfı az olduğu için transmission'a göre stratified.

    Regresyon ve kümeleme de aynı indeksleri kullanır; böylece hiçbir test satırı başka bir görevde eğitime girmez.
    """
    return train_test_split(df, test_size=TEST_SIZE, stratify=df[CLF_TARGET], random_state=SEED)
