import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_reliability_diagram(
    ece: float, centers: np.ndarray, widths: np.ndarray, conf: np.ndarray, acc: np.ndarray, counts: np.ndarray,
    n_bins: int, title: str = None, figsize: tuple[float, float] = (4,4), percent: bool = True, save_path: str = None
):
    """
    Reliability diagram with the following elements:
      - blue bars = empirical accuracy per bin
      - hatched red 'Calibration Error' rectangles = |acc - conf| per bin
      - diagonal y=x
      - ECE box (percent by default)

    Reference: https://arxiv.org/abs/1706.04599
    """
    # for plotting, replace NaNs (empty bins) by zeros for bars
    acc_plot = np.nan_to_num(acc, nan=0.0)

    fig, ax = plt.subplots(figsize=figsize)

    # Bars: accuracy per bin
    bars = ax.bar(
        centers, acc_plot, width=widths, align="center",
        edgecolor="black", alpha=0.7, label="Binned Accuracy", zorder=2
    )

    # Hatched gap rectangles per bin
    for c, w, a, cf, cnt in zip(centers, widths, acc, conf, counts):
        if cnt == 0 or np.isnan(a) or np.isnan(cf):
            continue
        lower = min(a, cf)
        height = abs(a - cf)
        if height == 0:
            continue
        rect = Rectangle(
            (c - w/2, lower), w, height,
            fill=False, hatch="///", edgecolor="tab:red", linewidth=0.0
        )
        # add a lightly transparent facecolor to make it visible
        rect_fc = Rectangle(
            (c - w/2, lower), w, height,
            facecolor="tab:red", alpha=0.15, edgecolor="none"
        )
        ax.add_patch(rect_fc)
        ax.add_patch(rect)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5)

    # Axes, grid, labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Probability", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

    # Legend entries (Binned Accuracy, Calibration Error)
    gap_proxy = Rectangle((0,0),1,1, fill=False, hatch="///", edgecolor="tab:red")
    ax.legend([bars, gap_proxy], ["Binned Accuracy", "Calibration Error"], loc="upper left", frameon=True, fontsize=12)

    # ECE box
    e_display = 100 * ece if percent else ece
    e_label = f"ECE={e_display:.2f}" + ("%" if percent else "")
    ax.text(
        0.8, 0.05, e_label,
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(facecolor="lightgray", alpha=0.9, boxstyle="round,pad=0.35")
    )

    if title is None:
        title = f"Uncal. – Reliability (bins={n_bins})"
    ax.set_title(title, fontsize=13)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    return fig, ax