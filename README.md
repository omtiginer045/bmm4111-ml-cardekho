# bmm4111-ml-cardekho

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omtiginer045/bmm4111-ml-cardekho/blob/main/notebooks/ML_F26_CarDekho.ipynb)

BMM4111 Makine Öğrenmesi dönem ödevi (Balıkesir Üniversitesi). Tek bir veri kümesi üzerinde üç problem:

| Görev | Hedef | Modeller |
|---|---|---|
| Sınıflandırma | `transmission` (Manual / Automatic) | Baseline, Random Forest, ANN (MLP) |
| Regresyon | `selling_price` (INR) | Baseline, Ridge, Random Forest |
| Kümeleme | — (hedefsiz) | KMeans |

Bir görevin hedefi diğer görevlerde özellik olarak kullanılmaz.

## Veri

- Kaynak: [Vehicle dataset — CarDekho (Kaggle, nehalbirla)](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho), dosya `Car details v3.csv`
- Lisans: Database: Open Database (ODbL), Contents: Database Contents (DbCL); yeniden dağıtıma izin verir.
- Repodaki kopya: `data/raw/car_details_v3.csv` (8128 satır × 13 sütun). Temizlik sonrası 6813 satır, 10 özellik.

## Kurulum

Python 3.10+ gerekir.

```bash
git clone https://github.com/omtiginer045/bmm4111-ml-cardekho.git
cd bmm4111-ml-cardekho
pip install -r requirements.txt
```

## Testler

```bash
pytest -q
```

Testler temizleme kurallarını, bölmenin tekrar üretilebilirliğini, ön işlemenin yalnız eğitimden öğrenildiğini ve
hedeflerin diğer görevlere sızmadığını denetler (tam grid search nedeniyle ~2 dk sürer).

## Notebook'u çalıştırma

- **Colab:** yukarıdaki rozete tıklayın ve *Çalışma zamanı → Tümünü çalıştır*. Defter depoyu kendisi klonlar.
- **Yerel:**

  ```bash
  jupyter notebook notebooks/ML_F26_CarDekho.ipynb
  # veya baştan sona komut satırından:
  jupyter nbconvert --to notebook --execute notebooks/ML_F26_CarDekho.ipynb --output run.ipynb
  ```

Defter tüm tabloları `reports/tables/*.csv`, tüm grafikleri `reports/figures/*.png` olarak kaydeder.
`SEED = 42` sabit olduğu için tablolar her çalıştırmada aynı çıkar.

## Klasör yapısı

```
├── data/raw/car_details_v3.csv     ham veri
├── docs/                           ödev metni ve uygulama planı
├── notebooks/ML_F26_CarDekho.ipynb uçtan uca defter (çıktılarıyla)
├── reports/figures/                grafikler
├── reports/tables/                 sonuç tabloları
├── src/cardekho/
│   ├── config.py                   SEED, yollar, hedefler, özellik listeleri
│   ├── data.py                     yükleme, birim parse, temizleme
│   ├── split.py                    tek stratified eğitim/test bölmesi
│   ├── preprocess.py               imputer + scaler + one-hot (Pipeline içinde)
│   ├── classification.py
│   ├── regression.py
│   └── clustering.py
└── tests/                          pytest testleri
```
