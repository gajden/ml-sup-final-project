from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_classes_pca(df: pd.DataFrame, y_col: str, random_state=42):
    """
    Visualize features spatial relations using PCA.
    :param df: source data.
    :param y_col: column name of target values.
    :return: None
    """
    X = df.drop(columns=[y_col])
    y = df[y_col]

    Xs = StandardScaler().fit_transform(X)

    pca2 = PCA(n_components=2, random_state=random_state)
    X_pca2 = pca2.fit_transform(Xs)

    classes = np.unique(y)
    n_classes = len(classes)
    cmap = get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(n_classes)]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.title("Dataset visualization with PCA")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, linestyle="--", alpha=0.3)

    for idx, cls in enumerate(classes):
        mask = (y == cls)
        ax.scatter(
            X_pca2[mask, 0],
            X_pca2[mask, 1],
            s=24,
            alpha=0.85,
            label=str(cls),
            c=[colors[idx]],
        )
    ax.legend(title="Class", markerscale=1.2, fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig, ax


def visualize_classes_tsne(
        df: pd.DataFrame,
        y_col: str,
        random_state: int = 42):
    """
    Visualize features spatial relations using TSNE.
    :param df: source data.
    :param y_col: column name of target values.
    :return: None
    """
    X = df.drop(columns=[y_col]).select_dtypes(include=[np.number]).copy()
    y = df[y_col]
    Xs = StandardScaler().fit_transform(X)
    n_samples, n_features = Xs.shape
    perplexity = n_samples // 3

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        verbose=0,
    )
    X_tsne = tsne.fit_transform(Xs)

    classes = np.unique(y.values)
    n_classes = len(classes)
    cmap = get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(n_classes)]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_title(f"Feature visualization t-SNE — perplexity={perplexity}")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(True, linestyle="--", alpha=0.3)

    for idx, cls in enumerate(classes):
        mask = (y.values == cls)
        ax.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            s=24,
            alpha=0.85,
            label=str(cls),
            c=[colors[idx]],
        )

    ax.legend(title="Class", markerscale=1.2, fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig, ax
