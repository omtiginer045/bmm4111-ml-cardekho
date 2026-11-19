# bmm4111-ml-cardekho

BMM4111 Makine Öğrenmesi dönem ödevi (Balıkesir Üniversitesi). Aynı veri kümesi üzerinde
sınıflandırma (`transmission`), regresyon (`selling_price`) ve kümeleme.

## Veri

- Kaynak: [Vehicle dataset — CarDekho (Kaggle, nehalbirla)](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho), dosya `Car details v3.csv`
- Lisans: Database: Open Database (ODbL), Contents: Database Contents (DbCL) — yeniden dağıtıma izin verir.
- Repodaki kopya: `data/raw/car_details_v3.csv` (8128 satır × 13 sütun)

## Kurulum ve testler

```bash
pip install -r requirements.txt
pytest -q
```
