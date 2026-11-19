import pandas as pd

from cardekho.clustering import (fit_kmeans, k_search, plot_k_search, plot_pca_clusters,
                                 profile_clusters)
from cardekho.config import CLF_TARGET, CLUSTER_FEATURES, REG_TARGET


def test_cluster_features_exclude_targets():
    assert CLF_TARGET not in CLUSTER_FEATURES and REG_TARGET not in CLUSTER_FEATURES


def test_k_search_shape_and_deterministic(sample_train_df):
    t1 = k_search(sample_train_df[CLUSTER_FEATURES]); t2 = k_search(sample_train_df[CLUSTER_FEATURES])
    assert list(t1["k"]) == list(range(2, 11))
    # KMeans'in paralel (OpenMP) toplaması inertia'nın son bitlerini değiştirebilir (~1e-16 göreli);
    # kümeler aynıdır, bu yüzden birebir eşitlik yerine çok dar tolerans.
    pd.testing.assert_frame_equal(t1, t2, rtol=1e-12)
    assert t1["inertia"].is_monotonic_decreasing
    assert t1["silhouette"].between(-1, 1).all()


def test_profile_has_one_row_per_cluster(sample_train_df):
    model = fit_kmeans(sample_train_df[CLUSTER_FEATURES], k=3)
    labels = model.predict(sample_train_df[CLUSTER_FEATURES])
    prof = profile_clusters(sample_train_df, labels)
    assert len(prof) == 3 and prof["n"].sum() == len(sample_train_df)
    assert {"automatic_ratio", "median_selling_price", "top_fuel", "top_brand"} <= set(prof.columns)


def test_plots_write_files(sample_train_df, tmp_path):
    X = sample_train_df[CLUSTER_FEATURES]
    plot_k_search(k_search(X, ks=range(2, 4)), tmp_path / "k.png")
    model = fit_kmeans(X, k=2)
    plot_pca_clusters(model.named_steps["pre"].transform(X), model.predict(X), tmp_path / "pca.png")
    assert (tmp_path / "k.png").exists() and (tmp_path / "pca.png").exists()
