import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_reliability_diagram(
    ece: float,
    centers: np.ndarray, widths: np.ndarray,
    conf: np.ndarray, acc: np.ndarray, counts: np.ndarray,
    n_bins: int,
    title: str = None,
    figsize: tuple[float, float] = (4,4),
    percent: bool = True,
    save_path: str = None,
):
    """
    Reliability diagram with:
      - blue bars = empirical accuracy per bin
      - hatched red 'Calibration Error' rectangles = |acc - conf| per bin
      - diagonal y=x
      - ECE box (percent by default)
      - (NEW) thin black bars on a secondary y-axis for % of predictions
    """
    show_pct = True
    pct_bar_frac = 0.18

    acc_plot = np.nan_to_num(acc, nan=0.0)
    conf_plot = np.nan_to_num(conf, nan=0.0)

    fig, ax = plt.subplots(figsize=figsize)

    # Accuracy bars
    bars = ax.bar(
        centers, acc_plot, width=widths, align="center",
        edgecolor="black", alpha=0.7, label="Binned Accuracy", zorder=3
    )

    # --- Hatched gap rectangles per bin (CLIPPED) ---
    for c, w, a, cf, cnt in zip(centers, widths, acc, conf, counts):
        if cnt == 0 or np.isnan(a) or np.isnan(cf):
            continue

        # clip horizontal extent to [0,1] (protects 1st/last bin visuals)
        x0 = max(c - w/2, 0.0)
        x1 = min(c + w/2, 1.0)
        if x1 <= x0:
            continue
        w_eff = x1 - x0

        # clip vertical extent to [0,1] BEFORE computing height
        lo = np.clip(min(a, cf), 0.0, 1.0)
        hi = np.clip(max(a, cf), 0.0, 1.0)
        height = max(0.0, hi - lo)
        if height == 0:
            continue

        # translucent fill + hatched outline
        rect_fc = Rectangle((x0, lo), w_eff, height,
                            facecolor="tab:red", alpha=0.3, edgecolor="none", zorder=4)
        rect = Rectangle((x0, lo), w_eff, height,
                         fill=False, hatch="///", edgecolor="tab:red", linewidth=0.0, zorder=5)
        ax.add_patch(rect_fc)
        ax.add_patch(rect)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5, zorder=2)

    # Axes, grid, labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Probability", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

    # --- Thin "% of predictions" bars on secondary y-axis (non-intrusive) ---
    ax2 = ax.twinx()

    total = max(int(counts.sum()), 1)
    pct = 100.0 * counts / total
    # thin bars centered in each bin; draw BEHIND accuracy bars
    thin_w = widths * pct_bar_frac
    ratio_bars = ax2.bar(centers, pct, width=thin_w, align="center",
            color="black", edgecolor="white", linewidth=0.8, alpha=0.8, zorder=1)
    ax2.set_ylim(0, max(5.0, float(pct.max()) * 1.15))
    ax2.set_ylabel("% of predictions", fontsize=11)
    ax2.tick_params(axis='y', labelsize=10)
    # keep x ticks from the main axis only
    ax2.set_yticks(ax2.get_yticks())  # ensure it draws ticks but doesn't steal layout

    # Legend (Binned Accuracy, Calibration Error)
    ax_xy = (0.01, 1.05)
    fig_xy = fig.transFigure.inverted().transform(ax.transAxes.transform(ax_xy))
    gap_proxy = Rectangle((0,0), 1,1, fill=False, hatch="///", edgecolor="tab:red")
    fig.legend([bars, gap_proxy, ratio_bars], ["Binned Accuracy", "Calibration Error", r"% of predictions"],
              loc="upper left", frameon=True, fontsize=12, bbox_to_anchor=fig_xy, bbox_transform=fig.transFigure)

    # ECE box (bottom-right)
    e_display = 100 * ece if percent else ece
    e_label = f"ECE={e_display:.2f}" + ("%" if percent else "")
    fig.text(
        0.97, 0.03, e_label,
        transform=ax.transAxes,
        fontsize=12,
        ha="right", va="bottom",
        bbox=dict(facecolor="lightgray", alpha=0.9, boxstyle="round,pad=0.35"),
        zorder=7
    )

    if title is None:
        title = f"Uncal. – Reliability (bins={n_bins})"
    ax.set_title(title, fontsize=13)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    return fig, ax