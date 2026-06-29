from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure


def plot_embedding_projector(
    embeddings: np.ndarray,
    labels: Any,
    save_path: str | Path | None = None,
    method: str = "tsne",
    figsize: tuple[int, int] = (10, 8),
    perplexity: float = 30.0,
    random_state: int = 42,
) -> Figure:
    import matplotlib.pyplot as plt

    n_samples = embeddings.shape[0]
    labels_arr = np.asarray(labels).ravel()
    unique_labels = sorted(np.unique(labels_arr))
    n_unique = len(unique_labels)

    if method == "tsne" and n_samples > 1:
        from sklearn.manifold import TSNE

        actual_perplexity = min(perplexity, n_samples - 1)
        projector = TSNE(
            n_components=2,
            perplexity=actual_perplexity,
            random_state=random_state,
        )
    else:
        from sklearn.decomposition import PCA

        projector = PCA(n_components=2, random_state=random_state)

    coords = projector.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.colormaps.get_cmap("tab10") if n_unique <= 10 else plt.colormaps.get_cmap("tab20")

    for i, label_val in enumerate(unique_labels):
        mask = labels_arr == label_val
        color = cmap(i % cmap.N)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[color],
            label=str(label_val),
            alpha=0.7,
            s=30,
            edgecolors="none",
        )

    method_name = "t-SNE" if method == "tsne" else "PCA"
    ax.set_title(f"Embedding Projection ({method_name})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.3)

    if n_unique <= 20:
        ax.legend(fontsize=8, markerscale=2)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig
