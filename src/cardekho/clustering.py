"""Kümeleme: hedefler olmadan, yalnız araç özellikleriyle doğal gruplar (KMeans)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from cardekho.config import CLF_TARGET, CLUSTER_FEATURES, REG_TARGET, SEED


def cluster_preprocessor() -> Pipeline:
    """KMeans uzaklık temelli: ölçek farkı (km_driven ~1e5, seats ~5) kümeleri tek sütuna boğmasın diye ölçeklenir.

    km_driven çok çarpık ve aykırı değerli; log1p ile sıkıştırılır. Sütun sırası: km_driven, sonra diğerleri.
    """
    return Pipeline([
        ("log", ColumnTransformer([("km", FunctionTransformer(np.log1p), ["km_driven"])],
                                  remainder="passthrough")),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def k_search(X: pd.DataFrame, ks=range(2, 11)) -> pd.DataFrame:
    """Her k için inertia (dirsek yöntemi) ve silhouette. Ön işleme yalnız verilen X'e (eğitim) fit edilir."""
    X_scaled = cluster_preprocessor().fit_transform(X[CLUSTER_FEATURES])
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_scaled)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X_scaled, km.labels_)})
    return pd.DataFrame(rows)


def fit_kmeans(X: pd.DataFrame, k: int) -> Pipeline:
    return Pipeline([
        ("pre", cluster_preprocessor()),
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=SEED)),
    ]).fit(X[CLUSTER_FEATURES])


def profile_clusters(df: pd.DataFrame, labels) -> pd.DataFrame:
    """Kümeleri alan anlamıyla yorumlamak için özet. Hedefler kümelemeye girmedi; burada yalnız sonradan yorum için."""
    g = df.assign(cluster=np.asarray(labels)).groupby("cluster")
    prof = g[CLUSTER_FEATURES].median()
    prof.insert(0, "n", g.size())
    prof["median_selling_price"] = g[REG_TARGET].median()
    prof["automatic_ratio"] = g[CLF_TARGET].apply(lambda s: (s == "Automatic").mean()).round(3)
    prof["top_fuel"] = g["fuel"].agg(lambda s: s.mode().iloc[0])
    prof["top_brand"] = g["brand"].agg(lambda s: s.mode().iloc[0])
    return prof.reset_index()


def plot_k_search(tbl: pd.DataFrame, path: Path) -> None:
    """Solda dirsek (inertia), sağda silhouette; k bu iki grafiğe birlikte bakılarak seçilir."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(tbl["k"], tbl["inertia"], "o-")
    ax1.set(xlabel="k", ylabel="Inertia", title="Dirsek yöntemi")
    ax2.plot(tbl["k"], tbl["silhouette"], "o-")
    ax2.set(xlabel="k", ylabel="Silhouette", title="Silhouette skoru")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def plot_pca_clusters(X_scaled, labels, path: Path) -> None:
    """Ölçeklenmiş 6 boyutlu veriyi görselleştirmek için PCA ile 2 boyuta indirir (yalnız görselleştirme)."""
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=8, alpha=0.6)
    ax.legend(*sc.legend_elements(), title="Küme")
    ev = pca.explained_variance_ratio_
    ax.set(xlabel=f"PC1 ({ev[0]:.0%})", ylabel=f"PC2 ({ev[1]:.0%})",
           title="Kümeler (PCA ile 2 boyut)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
