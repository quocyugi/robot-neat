"""
Statistical Analysis — Mann-Whitney U, Bonferroni, Boxplots
=============================================================
Tools for rigorous statistical comparison of experiment results.

Reference
---------
- brainstorm/04_experiments/statistical_methods.md (Decisions #28, #29)
- overview.md §6.3
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def mann_whitney_u(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Two-sided Mann-Whitney U test.

    Returns
    -------
    dict with keys: U_stat, p_value, significant, effect_size_r
    """
    u_stat, p_value = stats.mannwhitneyu(
        scores_a, scores_b, alternative="two-sided",
    )
    n = len(scores_a) + len(scores_b)
    # Effect size r = Z / sqrt(N)
    z = stats.norm.ppf(1 - p_value / 2) if p_value < 1.0 else 0.0
    r = abs(z) / np.sqrt(n) if n > 0 else 0.0

    return {
        "U_stat": float(u_stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "effect_size_r": float(r),
    }


def pairwise_tests(
    results: dict[str, np.ndarray],
    alpha: float = 0.05,
) -> list[dict]:
    """Run Mann-Whitney U between all pairs with Bonferroni correction.

    Parameters
    ----------
    results : dict mapping label → array of scores (e.g. 10 seeds).

    Returns
    -------
    list of dicts with comparison results.
    """
    labels = list(results.keys())
    n_comparisons = len(labels) * (len(labels) - 1) // 2
    corrected_alpha = alpha / max(n_comparisons, 1)

    comparisons = []
    for a, b in combinations(labels, 2):
        test = mann_whitney_u(results[a], results[b], alpha=corrected_alpha)
        test["comparison"] = f"{a} vs {b}"
        test["corrected_alpha"] = corrected_alpha
        comparisons.append(test)

    return comparisons


def plot_boxplot(
    results: dict[str, np.ndarray],
    metric_name: str = "Success Rate",
    output_path: Optional[str | Path] = None,
):
    """Generate a publication-quality boxplot comparing methods.

    Uses Median + IQR display (Decision #28).
    """
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    labels = []
    for label, scores in results.items():
        data.extend(scores.tolist())
        labels.extend([label] * len(scores))

    import pandas as pd
    df = pd.DataFrame({"Method": labels, metric_name: data})

    sns.boxplot(
        data=df, x="Method", y=metric_name,
        palette="Set2", showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 6},
        ax=ax,
    )
    ax.set_title(f"{metric_name} — 10 Independent Seeds", fontweight="bold")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
