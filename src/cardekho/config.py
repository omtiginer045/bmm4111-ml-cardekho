"""Projenin tüm sabitleri: seed, yollar, hedefler ve özellik listeleri."""
from pathlib import Path

SEED = 42
TEST_SIZE = 0.2

# Yollar depo köküne göre çözülür; notebook nereden çalışırsa çalışsın aynı dosyaları bulur.
ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "car_details_v3.csv"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"

CLF_TARGET = "transmission"
REG_TARGET = "selling_price"

# Hedefler (transmission, selling_price) bilerek bu listelerde yok.
NUM_FEATURES = ["year", "km_driven", "mileage", "engine", "max_power", "seats", "owner"]
CAT_FEATURES = ["fuel", "seller_type", "brand"]
FEATURES = NUM_FEATURES + CAT_FEATURES

CLUSTER_FEATURES = ["year", "km_driven", "mileage", "engine", "max_power", "seats"]
