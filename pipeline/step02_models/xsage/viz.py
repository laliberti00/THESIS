"""Plotting helpers for the X-SAGE artifacts in §6 of the brief.

Pure matplotlib + NumPy (one import each). Each function writes one PNG and
keeps inputs flat arrays; the orchestrator pairs these with CSV dumps.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# 6.1 clusters in 2D
# ---------------------------------------------------------------------------

def plot_clusters_2d(X2: np.ndarray,
                       core_label: np.ndarray,
                       is_boundary: np.ndarray,
                       prototypes_2d: np.ndarray | None,
                       out_path: Path,
                       title: str = "X-SAGE situations (PCA of v)"):
    """Filled dots = core, hollow ×s = boundary. Prototypes marked if given."""
    plt = _setup()
    fig, ax = plt.subplots(figsize=(7, 5))
    K = int(core_label.max() + 1)
    cmap = plt.cm.tab10
    for k in range(K):
        # core
        c = (core_label == k) & ~is_boundary
        if c.any():
            ax.scatter(X2[c, 0], X2[c, 1], s=18,
                        color=cmap(k % 10), label=f"sit {k} core", alpha=0.7)
        # boundary
        b = (core_label == k) & is_boundary
        if b.any():
            ax.scatter(X2[b, 0], X2[b, 1], s=22, marker="x",
                        color=cmap(k % 10), alpha=0.6)
    if prototypes_2d is not None:
        ax.scatter(prototypes_2d[:, 0], prototypes_2d[:, 1], s=160,
                    marker="*", edgecolor="black", color="white", linewidths=1.0,
                    label="prototypes")
    ax.set_title(title)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6.2 transition heatmap
# ---------------------------------------------------------------------------

def plot_transition_heatmap(T: np.ndarray, out_path: Path,
                              title: str = "Situation transition T") -> None:
    plt = _setup()
    K = T.shape[0]
    fig, ax = plt.subplots(figsize=(0.7 + 0.5 * K, 0.7 + 0.5 * K))
    im = ax.imshow(T, cmap="Blues", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center",
                     color="black" if T[i, j] < 0.5 else "white", fontsize=8)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([f"s{j}" for j in range(K)])
    ax.set_yticklabels([f"s{i}" for i in range(K)])
    ax.set_xlabel("next"); ax.set_ylabel("current")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6.2 dynamic fairness curve
# ---------------------------------------------------------------------------

def plot_dynamic_fairness(LT_curve: np.ndarray, out_path: Path,
                            title: str = "Dynamic fairness LT̄ vs τ") -> None:
    """LT_curve shape: (K, tau_max). One line per starting situation."""
    plt = _setup()
    K, tau_max = LT_curve.shape
    fig, ax = plt.subplots(figsize=(7, 4))
    taus = np.arange(1, tau_max + 1)
    for k in range(K):
        ax.plot(taus, LT_curve[k], marker="o", label=f"start s{k}")
    ax.set_xlabel("τ (steps ahead)")
    ax.set_ylabel("expected LT (long-tail share of top-K)")
    ax.set_title(title)
    ax.set_xticks(taus)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6.3 modulation summary
# ---------------------------------------------------------------------------

def plot_modulation_summary(kappa_axis: np.ndarray,
                              frac_changed_core: np.ndarray,
                              frac_changed_boundary: np.ndarray,
                              out_path: Path,
                              title: str = "Modulation frequency by κ_S") -> None:
    """Bar chart, x = κ_S grid; bars for core vs boundary."""
    plt = _setup()
    x = np.arange(len(kappa_axis))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, frac_changed_core, width=w, label="core requests")
    ax.bar(x + w/2, frac_changed_boundary, width=w, label="boundary requests")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k:g}" for k in kappa_axis])
    ax.set_xlabel("κ_S (situational confidence weight)")
    ax.set_ylabel("fraction of requests with changed top-K vs B_blind")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
