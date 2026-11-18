# BMM4111 Dönem Ödevi (CarDekho) — Uygulama Planı

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Kod yazarken `karpathy-guidelines` ilkelerine uy: en az kod, istenmeyen özellik yok, her adım doğrulanabilir.

**Goal:** CarDekho kullanılmış araç verisi üzerinde sınıflandırma (`transmission`), regresyon (`selling_price`) ve kümelemeyi sızıntısız, tekrar üretilebilir şekilde yapan; Colab'da baştan sona çalışan bir GitHub deposu.

**Architecture:** Test edilebilir mantık küçük bir `src/cardekho/` paketinde (temizleme, bölme, ön işleme, üç görev). `notebooks/ML_F26_CarDekho.ipynb` yalnızca bu fonksiyonları çağırır; EDA'yı yapar, tablo ve grafikleri `reports/` altına yazar. Tüm ön işleme sklearn `Pipeline` içinde olduğu için CV ve test sırasında yalnızca eğitimden fit edilir.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn ≥1.3, matplotlib, seaborn, pytest, Jupyter/Colab.

**Spec:** `docs/BMM4111_Donem_Odevi_f26.pdf` (hocanın ödev metni; Task 1'de repoya kopyalanır).

## Global Constraints

- Veri kümesi: CarDekho "Vehicle dataset" → `Car details v3.csv` (Kaggle, nehalbirla). Kaynak README'de ve raporda yazılacak.
- Ödev şartı: ≥8 özellik, ≥1000 gözlem, sayısal + kategorik değişkenler.
- **Bir görevin hedefi diğer görevlerde özellik olamaz:** sınıflandırma X'inde `selling_price` yok; regresyon X'inde `transmission` yok; kümelemede ikisi de yok.
- Tek bölme: `test_size=0.2`, `stratify=transmission`, `random_state=42`. Regresyon da **aynı** indeksleri kullanır.
- Test kümesi yalnızca son değerlendirmede kullanılır. Model ve ayar seçimi eğitim üzerinde 5 katlı CV ile yapılır.
- `SEED = 42` her yerde kullanılır (split, CV shuffle, modeller, KMeans).
- Sınıflandırmada en az 2 algoritma, biri ANN (`MLPClassifier`) olacak. ANN için en az bir mimari/hiperparametre denemesi yapılıp sonuç tablosu üretilecek.
- Sınıflandırma metrikleri: confusion matrix, accuracy, precision, recall, F1 (macro). Birincil ölçüt `f1_macro`.
- Regresyon metrikleri: MAE, MSE, RMSE, R², MAPE (fiyat hiçbir zaman 0'a yakın değil, bu yüzden MAPE dahil). Birincil ölçüt RMSE. Gerçek–tahmin grafiği çizilecek.
- Kümeleme: ölçekleme + KMeans; k dirsek yöntemi **ve** silhouette ile seçilir; kümeler alan anlamıyla profillenir.
- Sonuç tabloları ödevdeki sütunlarla birebir üretilir: Sınıflandırma `Model | CV | Eğitim | Test | Yorum`; Regresyon `Model | CV | MAE | MSE | RMSE | R² | MAPE | Yorum` (hücreler "eğitim / test"). "Yorum" sütunu boş bırakılır; öğrenci el yazısıyla dolduracak.
- Kod, yorum satırları ve README Türkçe; değişken ve fonksiyon adları İngilizce.
- Depo adı kısa: `bmm4111-ml-cardekho`. Notebook Colab'da `git clone` ile açılıp baştan sona çalışmalı.

## Veri Kararları (Task 2'nin gerekçesi; savunmada sorulur)

Bilinen ham sütunlar: `name, year, selling_price, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, torque, seats` (8128 satır; Task 1'de doğrulanır).

| Karar | Gerekçe |
|---|---|
| Ham veride tam tekrarlanan satırlar **bölmeden önce** silinir | Aynı satır hem train'de hem test'te olursa test skoru şişer (sızıntı senaryosu #1) |
| `name` → `brand` (ilk kelime); `name` atılır | ~2000 benzersiz model adı aşırı kardinalite yaratır, marka ise anlamlı bir özellik |
| `torque` atılır | Birimler karışık (Nm/kgm, rpm aralıkları); güvenilir parse edilemiyor, bilgisi de `max_power`/`engine` ile örtüşüyor |
| CNG ve LPG satırları atılır | `mileage` bu yakıtlarda km/kg, diğerlerinde kmpl; birimler karşılaştırılamaz ve bu satırlar çok az (~%1) |
| `mileage`, `engine`, `max_power` metinden sayıya çevrilir; boş veya `0` değerler NaN olur | 0 km/l veya 0 bhp fiziksel olarak geçersiz; NaN yapılıp pipeline'da imputer ile doldurulur |
| `owner` sıralı kodlanır: "Test Drive Car"=0, "First Owner"=1, "Second Owner"=2, "Third Owner"=3, "Fourth & Above Owner"=4 (ham veride değerler tam bu yazımla geçer) | Doğal bir sırası var |
| Sütun seçiminden sonra `drop_duplicates()` **bir kez daha** çalıştırılır | Ham veride yalnız model adı / torque yazımı farklı olan 19 ilan var; `name` ve `torque` atılınca birebir aynı oluyorlar ve 3'ü train ile test arasında bölünüyor (sızıntı) |
| Eksik değerler temizlemede **doldurulmaz** | Medyan/mod yalnızca eğitimden öğrenilmeli, bu yüzden imputer pipeline içinde |
| Aykırı `km_driven` değerleri silinmez, EDA'da raporlanır | Gerçek olabilirler; ağaç modelleri dayanıklı. Kümelemede `log1p` uygulanır |

Son özellik seti (10): sayısal `year, km_driven, mileage, engine, max_power, seats, owner`; kategorik `fuel, seller_type, brand`.

## Çalışma Düzeni (Claude Code)

- **Her Task bittiğinde:** testler yeşil → commit → `git push` → kullanıcıya kısa özet → **kullanıcının onayını bekle**. Onay gelmeden sonraki Task'a geçme (bu kural `executing-plans`'ın "durmadan devam et" varsayılanını geçersiz kılar).
- Özet şunları içerir: ne yapıldı, hangi testler eklendi, savunmada sorulabilecek 1–2 "neden" sorusu.

## İlk Denemeden Öğrenilenler (aynı hatalara düşme)

- Kaggle girişi yoksa CSV için GitHub aynası: `https://raw.githubusercontent.com/Vrundashah604/car-price-predictor/HEAD/data/Car%20details%20v3.csv` (8128×13; 1202 tam tekrar; Diesel 4402 / Petrol 3631 / CNG 57 / LPG 38; Manual 7078 / Automatic 1050). README'de kaynak olarak Kaggle gösterilir.
- Beklenen temizleme sonucu: 8128 → 6926 (tam tekrar) → 6832 (CNG/LPG) → **6813** (ad/torque farkıyla tekrar).
- `train_test_split` test boyutunu **yukarı** yuvarlar (`math.ceil`), testte `round` kullanma.
- Modül içinde `matplotlib.use("Agg")` **yapma**; notebook'ta grafikler görünmez olur.
- `pytest` ilk çalıştırmada eksik paket varsa: `pip install -r requirements.txt` (nbformat/nbconvert/ipykernel dahil).

## Review Focus

1. `max_power` hücresi boş, `" bhp"` veya `"0"` olabilir → NaN olmalı, program çökmemeli.
2. `mileage = "0.0 kmpl"` → NaN olmalı.
3. Test kümesinde eğitimde hiç görülmemiş bir marka olabilir → `OneHotEncoder(handle_unknown="infrequent_if_exist")` hata vermemeli.
4. Hedef sızıntısı: `transmission` regresyon X'ine veya `selling_price` sınıflandırma X'ine girerse sessizce iyi skor çıkar → testle engellenmeli.
5. `seats`, `engine` gibi NaN içeren satırlar ve çok az gözlemli `owner=Test Drive Car` → imputer ile doldurulmalı; CV katmanlarında hata çıkmamalı.

---

## Dosya Yapısı

```
bmm4111-ml-cardekho/
├── README.md                     # veri kaynağı, kurulum, çalıştırma, Colab linki
├── requirements.txt
├── pyproject.toml                # yalnızca pytest pythonpath ayarı
├── data/raw/car_details_v3.csv
├── docs/BMM4111_Donem_Odevi_f26.pdf
├── docs/superpowers/plans/2026-11-14-cardekho-ml-odevi.md
├── src/cardekho/
│   ├── __init__.py
│   ├── config.py                 # SEED, yollar, hedefler, özellik listeleri
│   ├── data.py                   # yükleme, parse, temizleme
│   ├── split.py                  # tek stratified bölme
│   ├── preprocess.py             # ColumnTransformer kurucu
│   ├── classification.py
│   ├── regression.py
│   └── clustering.py
├── tests/
│   ├── conftest.py               # küçük sahte DataFrame fixture'ı
│   ├── test_data.py
│   ├── test_split_preprocess.py
│   ├── test_classification.py
│   ├── test_regression.py
│   └── test_clustering.py
├── notebooks/ML_F26_CarDekho.ipynb
└── reports/figures/  reports/tables/
```

---

### Task 1: Repo iskeleti, veri ve config

**Files:**
- Create: `README.md` (iskelet), `requirements.txt`, `pyproject.toml`, `.gitignore`, `src/cardekho/__init__.py`, `src/cardekho/config.py`, `tests/test_data.py`
- Add: `data/raw/car_details_v3.csv`, `docs/BMM4111_Donem_Odevi_f26.pdf`

**Interfaces:**
- Produces (`config.py`): `SEED = 42`, `TEST_SIZE = 0.2`, `RAW_PATH: Path`, `CLF_TARGET = "transmission"`, `REG_TARGET = "selling_price"`, `NUM_FEATURES: list[str]`, `CAT_FEATURES: list[str]`, `FEATURES = NUM_FEATURES + CAT_FEATURES`, `CLUSTER_FEATURES = ["year","km_driven","mileage","engine","max_power","seats"]`, `FIG_DIR`, `TABLE_DIR`.
- Produces (`data.py`): `load_raw(path: Path = RAW_PATH) -> pd.DataFrame`

- [ ] **Step 1:** `git init`, klasörleri oluştur. Kaggle'dan CSV'yi indir ve `data/raw/car_details_v3.csv` olarak kaydet. Lisansı kontrol et; CSV'nin repoya konması uygun değilse README'ye indirme adımını yaz ve `.gitignore`'a ekle. Ödev PDF'ini `docs/` altına koy.
- [ ] **Step 2: Failing test**

```python
def test_raw_shape_and_columns():
    df = load_raw()
    assert df.shape == (8128, 13)
    assert list(df.columns) == ["name","year","selling_price","km_driven","fuel",
        "seller_type","transmission","owner","mileage","engine","max_power","torque","seats"]
```

- [ ] **Step 3:** `pytest tests/test_data.py -v` → FAIL (import hatası).
- [ ] **Step 4:** `config.py` ve `load_raw` yaz (`pd.read_csv`). `requirements.txt`: `pandas numpy scikit-learn>=1.3 matplotlib seaborn pytest jupyter`. `pyproject.toml`: `[tool.pytest.ini_options] pythonpath = ["src"]`.
- [ ] **Step 5:** Test PASS. Shape veya sütunlar farklı çıkarsa **dur ve kullanıcıya bildir**; plandaki veri kararları bu sütunlara dayanıyor.
- [ ] **Step 6:** `git commit -m "chore: repo iskeleti, ham veri ve config"`

---

### Task 2: Parse ve temizleme

**Files:** Modify `src/cardekho/data.py`; Create `tests/conftest.py`; Modify `tests/test_data.py`

**Interfaces:**
- Produces: `parse_number(s: pd.Series) -> pd.Series` (float; baştaki sayıyı alır, sayı yoksa veya 0 ise NaN)
- Produces: `clean(raw: pd.DataFrame) -> pd.DataFrame` (sütunlar: `FEATURES + [CLF_TARGET, REG_TARGET]`; index sıfırlanmış)
- `conftest.py` → fixture `raw_small`: Review Focus 1, 2 ve 5'teki durumları, bir tekrar satırını ve bir CNG satırını içeren ~10 satırlık sahte ham DataFrame.

- [ ] **Step 1: Failing tests**

```python
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
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Uygula. Sıra şöyle: `drop_duplicates()` (ham satırda) → CNG/LPG satırlarını at → `brand = name.str.split().str[0]` → `parse_number` ile mileage/engine/max_power → owner eşlemesi → sütun seçimi. `parse_number` için: `s.astype(str).str.extract(r"([\d.]+)")[0].astype(float)`, ardından 0 değerleri NaN yap.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** `git commit -m "feat: veri temizleme ve birim parse"`

---

### Task 3: Bölme ve ön işleme (sızıntı güvenceleri)

**Files:** Create `src/cardekho/split.py`, `src/cardekho/preprocess.py`, `tests/test_split_preprocess.py`

**Interfaces:**
- Produces: `make_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]` (train, test; `stratify=df[CLF_TARGET]`, `SEED`, `TEST_SIZE`)
- Produces: `build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer`. Fit **edilmemiş** döner. Sayısal sütunlar: `SimpleImputer(strategy="median")` + `StandardScaler()`. Kategorik sütunlar: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)`.

- [ ] **Step 1: Failing tests**

```python
def test_split_is_deterministic_and_stratified():
    df = clean(load_raw())
    tr1, te1 = make_split(df); tr2, te2 = make_split(df)
    assert tr1.index.equals(tr2.index) and te1.index.equals(te2.index)
    assert set(tr1.index).isdisjoint(te1.index)
    p_tr = (tr1[CLF_TARGET] == "Automatic").mean(); p_te = (te1[CLF_TARGET] == "Automatic").mean()
    assert abs(p_tr - p_te) < 0.01

def test_preprocessor_unseen_brand_and_nan(raw_small):
    df = clean(raw_small)
    pre = build_preprocessor(NUM_FEATURES, CAT_FEATURES).fit(df[FEATURES])
    new = df[FEATURES].head(1).copy(); new["brand"] = "Bilinmeyen"; new["max_power"] = np.nan
    out = pre.transform(new)
    assert not np.isnan(out).any()

def test_scaler_learns_only_from_train():
    df = clean(load_raw()); tr, te = make_split(df)
    pre = build_preprocessor(NUM_FEATURES, []).fit(tr[NUM_FEATURES])
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    assert np.allclose(scaler.mean_[0], tr["year"].mean())
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Uygula (pipeline adımları: `"impute"`, `"scale"`, `"onehot"`; transformer adları `"num"`, `"cat"`). **Step 4:** Run → PASS.
- [ ] **Step 5:** `git commit -m "feat: stratified bölme ve sızıntısız ön işleme"`

---

### Task 4: Sınıflandırma (`transmission`)

**Files:** Create `src/cardekho/classification.py`, `tests/test_classification.py`

**Interfaces:**
- Consumes: `build_preprocessor`, `FEATURES`, `NUM_FEATURES`, `CAT_FEATURES`, `SEED`
- Produces: `get_xy_clf(df) -> tuple[pd.DataFrame, pd.Series]` (X = `df[FEATURES]`; y = `(transmission == "Automatic").astype(int)`, yani pozitif sınıf Automatic)
- Produces: `build_clf_searches() -> dict[str, GridSearchCV]`. Anahtarlar `"Baseline"`, `"Random Forest"`, `"ANN (MLP)"`. Her biri `Pipeline([("pre", build_preprocessor(...)), ("model", ...)])`; `cv=StratifiedKFold(5, shuffle=True, random_state=SEED)`, `scoring="f1_macro"`, `return_train_score=True`.
  - Baseline: `DummyClassifier(strategy="most_frequent")`, boş grid
  - Random Forest: `RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)`; grid `max_depth: [None, 10, 20]`, `min_samples_leaf: [1, 5]`, `class_weight: [None, "balanced"]`
  - ANN: `MLPClassifier(max_iter=500, early_stopping=True, random_state=SEED)`; grid `hidden_layer_sizes: [(16,), (64,), (64, 32)]`, `alpha: [1e-4, 1e-2]`
- Produces: `evaluate_clf(search, X_tr, y_tr, X_te, y_te) -> dict` (anahtarlar `cv_f1`, `train_f1`, `test_f1`, `test_accuracy`, `test_precision`, `test_recall`, `confusion_matrix`; precision, recall ve F1 `average="macro"`)
- Produces: `clf_results_table(results: dict[str, dict]) -> pd.DataFrame` (sütunlar `["Model","CV","Eğitim","Test","Yorum"]`, Yorum boş)
- Produces: `mlp_experiment_table(search: GridSearchCV) -> pd.DataFrame` (her mimari/alpha için ortalama train ve CV F1; ANN yorumu bundan yapılacak)

- [ ] **Step 1: Failing tests** (`raw_small` yerine gerçek veriden 1500 satırlık stratified örneklem; hızlı olsun diye grid'ler küçük tutulur)

```python
def test_target_not_in_clf_features():
    X, y = get_xy_clf(clean(load_raw()))
    assert REG_TARGET not in X.columns and CLF_TARGET not in X.columns
    assert set(y.unique()) == {0, 1}

def test_models_beat_baseline(sample_split):
    res = {k: evaluate_clf(s, *sample_split) for k, s in build_clf_searches().items()}
    assert res["Random Forest"]["test_f1"] > res["Baseline"]["test_f1"]
    assert res["ANN (MLP)"]["test_f1"] > res["Baseline"]["test_f1"]

def test_results_table_columns(sample_split):
    ...
    assert list(tbl.columns) == ["Model","CV","Eğitim","Test","Yorum"] and len(tbl) == 3
```

`conftest.py`'ye üç fixture eklenir. Hepsi `clean(load_raw())` → `make_split` → her iki kümeden SEED ile 1500/400 satır örneklem alarak başlar:
- `sample_split`: `get_xy_clf` → `(X_tr, y_tr, X_te, y_te)`
- `sample_split_reg`: aynı örneklem + `get_xy_reg` (Task 5)
- `sample_train_df`: örneklenmiş train DataFrame'i (Task 6)

`config.py`'de `NUM_FEATURES` listesi `"year"` ile başlar (Task 3 testi buna dayanıyor).

- [ ] **Step 2:** Run → FAIL. **Step 3:** Uygula. **Step 4:** Run → PASS.
- [ ] **Step 5:** `git commit -m "feat: sınıflandırma (baseline, RF, MLP)"`

---

### Task 5: Regresyon (`selling_price`)

**Files:** Create `src/cardekho/regression.py`, `tests/test_regression.py`

**Interfaces:**
- Produces: `get_xy_reg(df) -> tuple[pd.DataFrame, pd.Series]` (X = `df[FEATURES]`, `transmission` yok; y = `selling_price`)
- Produces: `build_reg_searches() -> dict[str, GridSearchCV]`. Anahtarlar `"Baseline"`, `"Ridge"`, `"Random Forest"`. `cv=KFold(5, shuffle=True, random_state=SEED)`, `scoring="neg_root_mean_squared_error"`, `return_train_score=True`.
  - Baseline: `DummyRegressor(strategy="mean")`
  - Ridge: `TransformedTargetRegressor(regressor=Pipeline(pre + Ridge()), func=np.log1p, inverse_func=np.expm1)`; grid `regressor__model__alpha: [0.1, 1, 10, 100]`. Log alınmasının gerekçesi: fiyat sağa çarpık.
  - Random Forest: aynı log sarmalayıcı + `RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)`; grid `max_depth: [None, 10, 20]`, `min_samples_leaf: [1, 5]`
- Produces: `evaluate_reg(search, X_tr, y_tr, X_te, y_te) -> dict` (`cv_rmse` ve train/test için `mae`, `mse`, `rmse`, `r2`, `mape`; tümü orijinal fiyat ölçeğinde)
- Produces: `reg_results_table(results) -> pd.DataFrame` (sütunlar `["Model","CV","MAE","MSE","RMSE","R²","MAPE","Yorum"]`, hücreler `"eğitim / test"` biçiminde)
- Produces: `plot_actual_vs_pred(y_true, y_pred, path: Path) -> None` (log-log eksen, y=x doğrusu)

- [ ] **Step 1: Failing tests**

```python
def test_transmission_not_in_reg_features():
    X, _ = get_xy_reg(clean(load_raw()))
    assert CLF_TARGET not in X.columns and REG_TARGET not in X.columns

def test_baseline_r2_near_zero_and_models_better(sample_split_reg):
    res = {k: evaluate_reg(s, *sample_split_reg) for k, s in build_reg_searches().items()}
    assert abs(res["Baseline"]["test"]["r2"]) < 0.05
    assert res["Random Forest"]["test"]["rmse"] < res["Baseline"]["test"]["rmse"]
    assert res["Ridge"]["test"]["r2"] > 0.5
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Uygula. **Step 4:** Run → PASS.
- [ ] **Step 5:** `git commit -m "feat: regresyon (baseline, Ridge, RF) ve gerçek-tahmin grafiği"`

---

### Task 6: Kümeleme

**Files:** Create `src/cardekho/clustering.py`, `tests/test_clustering.py`

**Interfaces:**
- Produces: `cluster_preprocessor() -> Pipeline` (`km_driven` için `log1p` uygulayan `ColumnTransformer` + medyan imputer + `StandardScaler`, yalnızca `CLUSTER_FEATURES`)
- Produces: `k_search(X: pd.DataFrame, ks=range(2, 11)) -> pd.DataFrame` (sütunlar `k, inertia, silhouette`; `KMeans(n_clusters=k, n_init=10, random_state=SEED)`; ön işleme yalnızca X'e, yani eğitim kümesine fit edilir)
- Produces: `fit_kmeans(X, k) -> Pipeline`
- Produces: `profile_clusters(df: pd.DataFrame, labels) -> pd.DataFrame` (küme başına satır sayısı, `CLUSTER_FEATURES` medyanları, medyan `selling_price`, `transmission == Automatic` oranı, en sık `fuel` ve `brand`; hedefler yalnızca **sonradan yorum** için kullanılır)
- Produces: `plot_k_search(tbl, path)`, `plot_pca_clusters(X_scaled, labels, path)`

- [ ] **Step 1: Failing tests**

```python
def test_cluster_features_exclude_targets():
    assert CLF_TARGET not in CLUSTER_FEATURES and REG_TARGET not in CLUSTER_FEATURES

def test_k_search_shape_and_deterministic(sample_train_df):
    t1 = k_search(sample_train_df[CLUSTER_FEATURES]); t2 = k_search(sample_train_df[CLUSTER_FEATURES])
    assert list(t1["k"]) == list(range(2, 11)) and t1.equals(t2)
    assert t1["inertia"].is_monotonic_decreasing
    assert t1["silhouette"].between(-1, 1).all()
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Uygula. **Step 4:** Run → PASS.
- [ ] **Step 5:** `git commit -m "feat: KMeans kümeleme, k seçimi ve profil"`

---

### Task 7: Notebook, rapor çıktıları ve README

**Files:** Create `notebooks/ML_F26_CarDekho.ipynb`; Modify `README.md`

**Interfaces:** Consumes Task 1–6'daki tüm public fonksiyonlar. Yeni mantık **yazılmaz**; notebook yalnızca çağırır, gösterir ve kaydeder.

Notebook bölümleri (raporun bölüm sırasıyla aynı):
0. Kurulum: Colab için `!git clone ...` + `sys.path.append("src")`; seed bilgisi
1. Veri tanıma: shape, `dtypes`, eksik değer tablosu, tekrar sayısı, geçersiz 0 değerleri, `describe()`, `km_driven`/`selling_price` dağılımları, `transmission` sınıf oranı, korelasyon ısı haritası. Temizlemede kaç satırın neden düştüğü tablo halinde gösterilir.
2. Sınıflandırma: tablo, MLP deney tablosu, confusion matrix görseli, `classification_report`
3. Regresyon: tablo, gerçek–tahmin grafiği
4. Kümeleme: dirsek + silhouette grafiği, seçilen k, PCA grafiği, profil tablosu
5. Sızıntı ve deney tasarımı: yapılan kontrollerin listesi (Global Constraints'ten)

- [ ] **Step 1:** Notebook'u yaz. Her tablo `reports/tables/*.csv`, her grafik `reports/figures/*.png` olarak kaydedilir.
- [ ] **Step 2: Doğrulama.** `jupyter nbconvert --to notebook --execute notebooks/ML_F26_CarDekho.ipynb --output /tmp/run1.ipynb` hatasız bitmeli. İkinci çalıştırmada `reports/tables/*.csv` bayt bayt aynı olmalı (`diff -r` ile tekrar üretilebilirlik kontrolü).
- [ ] **Step 3:** README'yi yaz: proje amacı, veri kaynağı linki ve lisans notu, kurulum (`pip install -r requirements.txt`), testler (`pytest`), notebook'u çalıştırma, "Open in Colab" rozeti, klasör yapısı.
- [ ] **Step 4:** `pytest -q` → tüm testler PASS.
- [ ] **Step 5:** `git commit -m "feat: uçtan uca notebook, rapor çıktıları ve README"`

---

### Task 8: Son doğrulama

- [ ] **Step 1:** Temiz bir ortamda (yeni venv veya Colab) repoyu klonla → `pip install -r requirements.txt` → `pytest -q` PASS → notebook baştan sona çalışır.
- [ ] **Step 2:** Spec kontrolü: ödev PDF'indeki 3.1/1–6, 3.2/1–5, 3.3/1–4, 3.4 ve 4.1 maddelerinin her biri için notebook'ta karşılık gelen hücreyi işaretle; eksik varsa ekle.
- [ ] **Step 3:** README ve notebook'taki `<KULLANICI_ADI>` yer tutucularını gerçek GitHub kullanıcı adıyla değiştir, push et. Colab rozetinin defteri gerçekten açtığını kontrol et.

---

## Kod Dışı İşler (öğrencinin kendisi yapar)

| Tarih | İş |
|---|---|
| 14–22 Kas | Task 1–3 (veri ve altyapı) |
| 23–29 Kas | Task 4–5 |
| 30 Kas–6 Ara | Task 6–8 |
| 7–13 Ara | El yazısı rapor: ilk sayfa bilgileri, veri tanıma, 3 görev, sızıntı, sonuç, GitHub linki; sayfa numaraları |
| 14–18 Ara | CamScanner → `OgrenciNo_Omer_Coskun_ML_F26.pdf` → Teams; asılları elden teslim (20 Ara pazar, hocaya teslim günü sorulacak); savunma provası |

Savunmada sorulabilecek sorular: Scaler neden yalnızca train'e fit edildi? Tekrar satırlar neden bölmeden önce silindi? Sınıflar dengesizken neden accuracy değil macro-F1? Regresyonda neden log hedef kullanıldı? k nasıl seçildi? `transmission` regresyonda neden yok?
