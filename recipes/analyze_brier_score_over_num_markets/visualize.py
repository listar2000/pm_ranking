"""
Visualization codes for analyzing the "dilution" effect in Brier scores.
Professional, publication-ready plots with consistent styling.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, Literal
from utils import (
    DEFAULT_NUM_MARKETS_KEY, DEFAULT_DILUTION_KEY, DEFAULT_BRIER_SCORE_KEY,
    FORECASTER_KEY, calc_num_markets_distribution, get_num_markets_by_forecaster,
    calc_avg_brier_score_over_num_markets, calc_avg_brier_score_by_forecaster_over_num_markets,
    get_dilution_num_markets_pairs, calc_dilution_num_markets_correlation,
    calc_avg_brier_score_over_dilution, calc_brier_dilution_by_forecaster,
    run_regression_brier_score_over_dilution, run_regression_comparison
)

# ============================================================================
# Global style settings
# ============================================================================
def set_style():
    """Set professional matplotlib/seaborn style for all plots."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

# Color palette for forecasters (colorblind-friendly)
FORECASTER_COLORS = {
    "anthropic/claude-opus-4.1": "#0077B6",  # Blue
    "x-ai/grok-4": "#E63946",                # Red
    "o3": "#2A9D8F",                         # Teal
    "gemini-2.0-flash": "#F4A261",           # Orange
}

def get_forecaster_color(forecaster: str) -> str:
    """Get color for a forecaster, with fallback to seaborn palette."""
    if forecaster in FORECASTER_COLORS:
        return FORECASTER_COLORS[forecaster]
    # Generate consistent color for unknown forecasters
    palette = sns.color_palette("husl", 8)
    return palette[hash(forecaster) % len(palette)]

def get_short_name(forecaster: str) -> str:
    """Get a shorter display name for forecasters."""
    name_map = {
        "anthropic/claude-opus-4.1": "Claude Opus 4.1",
        "x-ai/grok-4": "Grok 4",
        "o3": "O3",
        "gemini-2.0-flash": "Gemini 2.0 Flash",
    }
    return name_map.get(forecaster, forecaster.split("/")[-1])


# ============================================================================
# Visualization codes for analysis part (i) - Market distribution
# ============================================================================
def plot_num_markets_histogram(
    df: pd.DataFrame,
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY,
    title: str = "Distribution of Number of Markets per Event",
    figsize: tuple = (10, 6),
    bins: int | str = "auto",
    x_lim = None,
    show_stats: bool = True,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot histogram of the number of markets per event (across all forecasters).
    
    Args:
        df: Cleaned predictions DataFrame
        num_markets_key: Column name for number of markets
        title: Plot title
        figsize: Figure size
        bins: Number of bins or 'auto'
        x_lim: Tuple of (x_min, x_max) to set the limits of the x-axis
        show_stats: Whether to show summary statistics box
        ax: Optional existing axes to plot on
        save_path: Path to save the figure
        
    Returns:
        Figure and Axes objects
    """
    set_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Get unique events to avoid counting duplicates
    event_df = df.drop_duplicates(subset=["event_ticker"])
    num_markets = event_df[num_markets_key]
    
    # Plot histogram
    ax.hist(num_markets, bins=bins, edgecolor='white', alpha=0.8, color='#0077B6')
    
    ax.set_xlabel("Number of Markets")
    ax.set_ylabel("Number of Events")
    if x_lim is not None:
        ax.set_xlim(x_lim)
    ax.set_title(title, fontweight='bold')
    
    if show_stats:
        stats_text = (
            f"n = {len(num_markets)}\n"
            f"Mean = {num_markets.mean():.1f}\n"
            f"Median = {num_markets.median():.0f}\n"
            f"Std = {num_markets.std():.1f}"
        )
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_num_markets_by_forecaster(
    df: pd.DataFrame,
    forecasters: Optional[list[str]] = None,
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY,
    plot_type: Literal["violin", "box", "strip"] = "violin",  # "violin", "box", or "strip"
    title: str = "Distribution of Markets by Forecaster",
    y_lim = None,
    figsize: tuple = (12, 6),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Compare the distribution of number of markets across different forecasters.
    
    Args:
        df: Cleaned predictions DataFrame
        forecasters: List of forecasters to include (None for all)
        num_markets_key: Column name for number of markets
        plot_type: Type of plot - "violin", "box", or "strip"
        title: Plot title
        y_lim: Tuple of (y_min, y_max) to set the limits of the y-axis
        figsize: Figure size
        ax: Optional existing axes
        save_path: Path to save the figure
        
    Returns:
        Figure and Axes objects
    """
    set_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    if forecasters is None:
        forecasters = df[FORECASTER_KEY].unique().tolist()
    
    # Prepare data
    plot_df = df[df[FORECASTER_KEY].isin(forecasters)].copy()
    plot_df["Forecaster"] = plot_df[FORECASTER_KEY].apply(get_short_name)
    
    # Order by median for better comparison
    order = (plot_df.groupby("Forecaster")[num_markets_key]
             .median().sort_values().index.tolist())
    
    # Get colors
    colors = [get_forecaster_color(f) for f in forecasters]
    palette = dict(zip([get_short_name(f) for f in forecasters], colors))
    
    if plot_type == "violin":
        sns.violinplot(data=plot_df, x="Forecaster", y=num_markets_key,
                       order=order, palette=palette, ax=ax, inner="box")
    elif plot_type == "box":
        sns.boxplot(data=plot_df, x="Forecaster", y=num_markets_key,
                    order=order, palette=palette, ax=ax)
    elif plot_type == "strip":
        sns.stripplot(data=plot_df, x="Forecaster", y=num_markets_key,
                      order=order, palette=palette, ax=ax, alpha=0.5, jitter=True)
    
    ax.set_xlabel("")
    ax.set_ylabel("Number of Markets")
    ax.set_title(title, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


# ============================================================================
# Visualization codes for analysis part (ii) - Brier score vs num_markets
# ============================================================================
def plot_brier_vs_num_markets(
    df: pd.DataFrame,
    forecaster: Optional[str] = None,
    normalize_by_event: bool = False,
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY,
    brier_score_key: str = DEFAULT_BRIER_SCORE_KEY,
    title: Optional[str] = None,
    show_error_bars: bool = True,
    x_lim = None,
    figsize: tuple = (10, 6),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot average Brier score vs number of markets (demonstrating dilution effect).
    
    Args:
        df: Cleaned predictions DataFrame
        forecaster: Specific forecaster (None for all)
        normalize_by_event: Whether to first average within events
        num_markets_key: Column name for number of markets
        brier_score_key: Column name for Brier score
        title: Plot title
        show_error_bars: Whether to show SEM error bars
        x_lim: Tuple of (x_min, x_max) to set the limits of the x-axis
        figsize: Figure size
        ax: Optional existing axes
        save_path: Path to save the figure
        
    Returns:
        Figure and Axes objects
    """
    set_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    stats_df = calc_avg_brier_score_over_num_markets(
        df, forecaster=forecaster, normalize_by_event=normalize_by_event,
        brier_score_key=brier_score_key, num_markets_key=num_markets_key
    )
    
    color = get_forecaster_color(forecaster) if forecaster else "#0077B6"
    
    # Plot line with markers
    ax.plot(stats_df[num_markets_key], stats_df["mean_brier"], 
            marker='o', color=color, linewidth=2, markersize=6)
    
    if show_error_bars:
        ax.fill_between(
            stats_df[num_markets_key],
            stats_df["mean_brier"] - stats_df["sem_brier"],
            stats_df["mean_brier"] + stats_df["sem_brier"],
            alpha=0.2, color=color
        )
    
    ax.set_xlabel("Number of Markets")
    ax.set_ylabel("Average Brier Score")
    
    if title is None:
        forecaster_str = get_short_name(forecaster) if forecaster else "All Forecasters"
        title = f"Brier Score vs. Number of Markets ({forecaster_str})"
    ax.set_title(title, fontweight='bold')
    
    if x_lim is not None:
        ax.set_xlim(x_lim)
    
    # Add annotation about trend
    if len(stats_df) > 1:
        x = stats_df[num_markets_key].values
        y = stats_df["mean_brier"].values
        slope = np.polyfit(x, y, 1)[0]
        trend = "↓ decreasing" if slope < 0 else "↑ increasing"
        ax.text(0.97, 0.97, f"Trend: {trend}\nSlope: {slope:.4f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_brier_vs_num_markets_comparison(
    df: pd.DataFrame,
    forecasters: Optional[list[str]] = None,
    normalize_by_event: bool = False,
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY,
    brier_score_key: str = DEFAULT_BRIER_SCORE_KEY,
    title: str = "Brier Score vs. Number of Markets by Forecaster",
    x_lim = None,
    show_error_bands: bool = True,
    figsize: tuple = (12, 7),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Compare Brier score vs number of markets across multiple forecasters.
    
    Args:
        df: Cleaned predictions DataFrame
        forecasters: List of forecasters (None for all)
        normalize_by_event: Whether to first average within events
        x_lim: Tuple of (x_min, x_max) to set the limits of the x-axis
        title: Plot title
        show_error_bands: Whether to show SEM error bands
        figsize: Figure size
        ax: Optional existing axes
        save_path: Path to save the figure
        
    Returns:
        Figure and Axes objects
    """
    set_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    if forecasters is None:
        forecasters = df[FORECASTER_KEY].unique().tolist()
    
    stats_dict = calc_avg_brier_score_by_forecaster_over_num_markets(
        df, forecasters=forecasters, normalize_by_event=normalize_by_event,
        brier_score_key=brier_score_key, num_markets_key=num_markets_key
    )
    
    for forecaster, stats_df in stats_dict.items():
        color = get_forecaster_color(forecaster)
        label = get_short_name(forecaster)
        
        ax.plot(stats_df[num_markets_key], stats_df["mean_brier"],
                marker='o', color=color, linewidth=2, markersize=5, label=label)
        
        if show_error_bands:
            ax.fill_between(
                stats_df[num_markets_key],
                stats_df["mean_brier"] - stats_df["sem_brier"],
                stats_df["mean_brier"] + stats_df["sem_brier"],
                alpha=0.15, color=color
            )
    
    ax.set_xlabel("Number of Markets")
    ax.set_ylabel("Average Brier Score")
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


# ============================================================================
# Visualization codes for analysis part (iii) - Dilution vs num_markets
# ============================================================================
def plot_dilution_vs_num_markets(
    df: pd.DataFrame,
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY,
    dilution_key: str = DEFAULT_DILUTION_KEY,
    title: str = "Dilution vs. Number of Markets",
    show_correlation: bool = True,
    show_trend_line: bool = True,
    use_hexbin: bool = False,
    figsize: tuple = (10, 7),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot of dilution (fraction of true outcomes) vs number of markets.
    This helps understand if events with more markets tend to have specific dilution patterns.
    
    Args:
        df: Cleaned predictions DataFrame
        num_markets_key: Column name for number of markets
        dilution_key: Column name for dilution
        title: Plot title
        show_correlation: Whether to annotate correlation statistics
        show_trend_line: Whether to show a trend line
        use_hexbin: Use hexbin instead of scatter for dense data
        figsize: Figure size
        ax: Optional existing axes
        save_path: Path to save the figure
        
    Returns:
        Figure and Axes objects
    """
    set_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    event_df = get_dilution_num_markets_pairs(df, dilution_key, num_markets_key)
    x = event_df[num_markets_key].values
    y = event_df[dilution_key].values
    
    if use_hexbin and len(event_df) > 200:
        hb = ax.hexbin(x, y, gridsize=20, cmap='Blues', mincnt=1)
        plt.colorbar(hb, ax=ax, label='Count')
    else:
        ax.scatter(x, y, alpha=0.6, edgecolor='white', s=50, c='#0077B6')
    
    if show_trend_line:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), '--', color='#E63946', linewidth=2, label='Trend line')
    
    ax.set_xlabel("Number of Markets")
    ax.set_ylabel("1 - Dilution (Fraction True)")
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    
    if show_correlation:
        corr_stats = calc_dilution_num_markets_correlation(df, dilution_key, num_markets_key)
        stats_text = (
            f"n = {corr_stats['n_events']}\n"
            f"Pearson r = {corr_stats['pearson_r']:.3f} (p={corr_stats['pearson_p']:.3e})\n"
            f"Spearman ρ = {corr_stats['spearman_r']:.3f} (p={corr_stats['spearman_p']:.3e})"
        )
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    if show_trend_line:
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_dilution_by_market_bins(
    df: pd.DataFrame,
    bins: list | int = [2, 3, 4, 5, 10, 20],
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY,
    dilution_key: str = DEFAULT_DILUTION_KEY,
    title: str = "Dilution Distribution by Market Count Bins",
    figsize: tuple = (12, 6),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Show dilution distribution grouped by number of markets (binned).
    Useful for seeing how dilution varies across different market counts.
    
    Args:
        df: Cleaned predictions DataFrame
        bins: Bin edges or number of bins
        title: Plot title
        figsize: Figure size
        ax: Optional existing axes
        save_path: Path to save the figure
        
    Returns:
        Figure and Axes objects
    """
    set_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    event_df = get_dilution_num_markets_pairs(df, dilution_key, num_markets_key)
    
    # Create bins
    if isinstance(bins, int):
        event_df["market_bin"] = pd.cut(event_df[num_markets_key], bins=bins)
    else:
        event_df["market_bin"] = pd.cut(event_df[num_markets_key], bins=bins + [np.inf])
    
    sns.boxplot(data=event_df, x="market_bin", y=dilution_key, ax=ax,
                palette="Blues", boxprops=dict(alpha=0.7))
    
    ax.set_xlabel("Number of Markets (bins)")
    ax.set_ylabel("1 - Dilution (Fraction True)")
    ax.set_title(title, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


# ============================================================================
# Visualization codes for analysis part (iv) - Regression analysis
# ============================================================================
def plot_brier_vs_dilution_with_regression(
    df: pd.DataFrame,
    forecaster: Optional[str] = None,
    include_num_markets: bool = True,
    brier_score_key: str = DEFAULT_BRIER_SCORE_KEY,
    dilution_key: str = DEFAULT_DILUTION_KEY,
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY,
    title: Optional[str] = None,
    show_ci: bool = True,
    figsize: tuple = (10, 7),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes, dict]:
    """
    Scatter plot of Brier score vs dilution with regression line and statistics.
    
    Args:
        df: Cleaned predictions DataFrame
        forecaster: Specific forecaster (None for all)
        include_num_markets: Include num_markets in regression model
        title: Plot title
        show_ci: Show 95% confidence interval band
        figsize: Figure size
        ax: Optional existing axes
        save_path: Path to save the figure
        
    Returns:
        Figure, Axes, and regression results dictionary
    """
    set_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Run regression
    reg_result = run_regression_brier_score_over_dilution(
        df, forecaster=forecaster, include_num_markets=include_num_markets,
        brier_score_key=brier_score_key, dilution_key=dilution_key,
        num_markets_key=num_markets_key
    )
    
    event_df = reg_result["data"]
    x = event_df["dilution"].values
    y = event_df["avg_brier"].values
    
    color = get_forecaster_color(forecaster) if forecaster else "#0077B6"
    
    # Scatter plot
    ax.scatter(x, y, alpha=0.5, edgecolor='white', s=40, c=color)
    
    # Simple OLS line for dilution (marginal effect)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, p(x_line), '-', color='#E63946', linewidth=2.5, label='Linear fit')
    
    if show_ci:
        # Bootstrap confidence interval for the line
        from scipy import stats
        n = len(x)
        y_pred = p(x)
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n - 2)
        se = np.sqrt(mse * (1/n + (x_line - x.mean())**2 / np.sum((x - x.mean())**2)))
        ci = 1.96 * se
        ax.fill_between(x_line, p(x_line) - ci, p(x_line) + ci, 
                        alpha=0.15, color='#E63946')
    
    ax.set_xlabel("1 - Dilution (Fraction True)")
    ax.set_ylabel("Average Brier Score (per event)")
    
    if title is None:
        forecaster_str = get_short_name(forecaster) if forecaster else "All Forecasters"
        title = f"Brier Score vs. Dilution ({forecaster_str})"
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    
    # Annotation box with regression results
    coef = reg_result["coefficients"]
    stats_text = (
        f"n = {reg_result['n_obs']}\n"
        f"R² = {reg_result['r_squared']:.3f}\n"
        f"Dilution β = {coef['dilution']['estimate']:.4f}\n"
        f"  (p = {coef['dilution']['pvalue']:.3e})"
    )
    if include_num_markets:
        stats_text += (
            f"\nNum Markets β = {coef['num_markets']['estimate']:.5f}\n"
            f"  (p = {coef['num_markets']['pvalue']:.3e})"
        )
    
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax.legend(loc='lower right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax, reg_result


def plot_brier_vs_dilution_comparison(
    df: pd.DataFrame,
    forecasters: Optional[list[str]] = None,
    brier_score_key: str = DEFAULT_BRIER_SCORE_KEY,
    dilution_key: str = DEFAULT_DILUTION_KEY,
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY,
    title: str = "Brier Score vs. Dilution by Forecaster",
    figsize: tuple = (12, 8),
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Compare Brier vs dilution relationship across multiple forecasters.
    
    Args:
        df: Cleaned predictions DataFrame
        forecasters: List of forecasters (None for all)
        title: Plot title
        figsize: Figure size
        ax: Optional existing axes
        save_path: Path to save the figure
        
    Returns:
        Figure and Axes objects
    """
    set_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    if forecasters is None:
        forecasters = df[FORECASTER_KEY].unique().tolist()
    
    dilution_data = calc_brier_dilution_by_forecaster(
        df, forecasters=forecasters, brier_score_key=brier_score_key,
        dilution_key=dilution_key, num_markets_key=num_markets_key
    )
    
    for forecaster, event_df in dilution_data.items():
        color = get_forecaster_color(forecaster)
        label = get_short_name(forecaster)
        
        x = event_df["dilution"].values
        y = event_df["avg_brier"].values
        
        # Scatter
        ax.scatter(x, y, alpha=0.4, s=30, c=color, label=f"{label} (n={len(x)})")
        
        # Trend line
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(0, 1, 100)
            ax.plot(x_line, p(x_line), '--', color=color, linewidth=2, alpha=0.8)
    
    ax.set_xlabel("1 - Dilution (Fraction True)")
    ax.set_ylabel("Average Brier Score (per event)")
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_regression_coefficients_comparison(
    df: pd.DataFrame,
    forecasters: Optional[list[str]] = None,
    include_num_markets: bool = True,
    brier_score_key: str = DEFAULT_BRIER_SCORE_KEY,
    dilution_key: str = DEFAULT_DILUTION_KEY,
    num_markets_key: str = DEFAULT_NUM_MARKETS_KEY,
    title: str = "Regression Coefficients Comparison",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Compare regression coefficients across forecasters using coefficient plots.
    
    Args:
        df: Cleaned predictions DataFrame
        forecasters: List of forecasters (None for all)
        include_num_markets: Include num_markets in regression
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Figure and Axes objects
    """
    set_style()
    
    if forecasters is None:
        forecasters = df[FORECASTER_KEY].unique().tolist()
    
    # Run regressions
    results = []
    for forecaster in forecasters:
        reg = run_regression_brier_score_over_dilution(
            df, forecaster=forecaster, include_num_markets=include_num_markets,
            brier_score_key=brier_score_key, dilution_key=dilution_key,
            num_markets_key=num_markets_key
        )
        coef = reg["coefficients"]
        results.append({
            "forecaster": get_short_name(forecaster),
            "dilution_coef": coef["dilution"]["estimate"],
            "dilution_ci_lower": coef["dilution"]["ci_lower"],
            "dilution_ci_upper": coef["dilution"]["ci_upper"],
            "dilution_sig": coef["dilution"]["pvalue"] < 0.05,
        })
        if include_num_markets:
            results[-1].update({
                "num_markets_coef": coef["num_markets"]["estimate"],
                "num_markets_ci_lower": coef["num_markets"]["ci_lower"],
                "num_markets_ci_upper": coef["num_markets"]["ci_upper"],
                "num_markets_sig": coef["num_markets"]["pvalue"] < 0.05,
            })
    
    results_df = pd.DataFrame(results)
    
    n_coefs = 2 if include_num_markets else 1
    fig, axes = plt.subplots(1, n_coefs, figsize=figsize)
    if n_coefs == 1:
        axes = [axes]
    
    # Dilution coefficient plot
    y_pos = np.arange(len(results_df))
    colors = [get_forecaster_color(f) for f in forecasters]
    
    ax = axes[0]
    for i, (_, row) in enumerate(results_df.iterrows()):
        marker = 'o' if row["dilution_sig"] else 's'
        ax.errorbar(row["dilution_coef"], i, 
                   xerr=[[row["dilution_coef"] - row["dilution_ci_lower"]], 
                         [row["dilution_ci_upper"] - row["dilution_coef"]]],
                   fmt=marker, color=colors[i], capsize=4, markersize=8)
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df["forecaster"])
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title("1 - Dilution Effect", fontweight='bold')
    
    if include_num_markets:
        ax = axes[1]
        for i, (_, row) in enumerate(results_df.iterrows()):
            marker = 'o' if row["num_markets_sig"] else 's'
            ax.errorbar(row["num_markets_coef"], i,
                       xerr=[[row["num_markets_coef"] - row["num_markets_ci_lower"]],
                             [row["num_markets_ci_upper"] - row["num_markets_coef"]]],
                       fmt=marker, color=colors[i], capsize=4, markersize=8)
        
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([])
        ax.set_xlabel("Coefficient (95% CI)")
        ax.set_title("Num Markets Effect", fontweight='bold')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    # Add legend for significance
    sig_patch = mpatches.Patch(color='gray', label='○ = p<0.05, □ = p≥0.05')
    fig.legend(handles=[sig_patch], loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_regression_summary_table(
    df: pd.DataFrame,
    forecasters: Optional[list[str]] = None,
    include_num_markets: bool = True,
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Create a summary table visualization of regression results across forecasters.
    
    Args:
        df: Cleaned predictions DataFrame
        forecasters: List of forecasters (None for all)
        include_num_markets: Include num_markets in regression
        figsize: Figure size
        save_path: Path to save the figure
        
    Returns:
        Figure, Axes, and summary DataFrame
    """
    set_style()
    
    summary_df = run_regression_comparison(
        df, forecasters=forecasters, include_num_markets=include_num_markets
    )
    
    # Format for display
    display_df = summary_df.copy()
    display_df["forecaster"] = display_df["forecaster"].apply(get_short_name)
    display_df["r_squared"] = display_df["r_squared"].apply(lambda x: f"{x:.3f}")
    display_df["dilution_coef"] = display_df["dilution_coef"].apply(lambda x: f"{x:.4f}")
    display_df["dilution_pvalue"] = display_df["dilution_pvalue"].apply(
        lambda x: f"{x:.3e}" if x < 0.001 else f"{x:.3f}"
    )
    if include_num_markets:
        display_df["num_markets_coef"] = display_df["num_markets_coef"].apply(lambda x: f"{x:.5f}")
        display_df["num_markets_pvalue"] = display_df["num_markets_pvalue"].apply(
            lambda x: f"{x:.3e}" if x < 0.001 else f"{x:.3f}"
        )
    
    # Rename columns for display
    col_names = {
        "forecaster": "Forecaster",
        "n_events": "N Events",
        "r_squared": "R²",
        "dilution_coef": "Dilution β",
        "dilution_pvalue": "Dilution p",
    }
    if include_num_markets:
        col_names.update({
            "num_markets_coef": "NumMkts β",
            "num_markets_pvalue": "NumMkts p",
        })
    display_df = display_df.rename(columns=col_names)
    display_df = display_df[[v for v in col_names.values()]]
    
    # Create table figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for j in range(len(display_df.columns)):
        table[(0, j)].set_facecolor('#0077B6')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title("Regression Results Summary", fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax, summary_df


# ============================================================================
# Convenience function for generating all plots at once
# ============================================================================
def generate_all_analysis_plots(
    df: pd.DataFrame,
    forecasters: Optional[list[str]] = None,
    output_dir: str = "./figures",
    prefix: str = "brier_dilution",
) -> dict:
    """
    Generate all analysis plots and save to output directory.
    
    Args:
        df: Cleaned predictions DataFrame
        forecasters: List of forecasters (None for all)
        output_dir: Directory to save figures
        prefix: Filename prefix
        
    Returns:
        Dictionary of figure paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {}
    
    # Part (i) plots
    fig, _ = plot_num_markets_histogram(df, save_path=f"{output_dir}/{prefix}_i_histogram.png")
    plt.close(fig)
    paths["i_histogram"] = f"{output_dir}/{prefix}_i_histogram.png"
    
    fig, _ = plot_num_markets_by_forecaster(df, forecasters=forecasters,
                                            save_path=f"{output_dir}/{prefix}_i_violin.png")
    plt.close(fig)
    paths["i_violin"] = f"{output_dir}/{prefix}_i_violin.png"
    
    # Part (ii) plots
    fig, _ = plot_brier_vs_num_markets_comparison(df, forecasters=forecasters,
                                                   save_path=f"{output_dir}/{prefix}_ii_comparison.png")
    plt.close(fig)
    paths["ii_comparison"] = f"{output_dir}/{prefix}_ii_comparison.png"
    
    # Part (iii) plots
    fig, _ = plot_dilution_vs_num_markets(df, save_path=f"{output_dir}/{prefix}_iii_scatter.png")
    plt.close(fig)
    paths["iii_scatter"] = f"{output_dir}/{prefix}_iii_scatter.png"
    
    # Part (iv) plots
    fig, _, _ = plot_brier_vs_dilution_with_regression(df, save_path=f"{output_dir}/{prefix}_iv_regression.png")
    plt.close(fig)
    paths["iv_regression"] = f"{output_dir}/{prefix}_iv_regression.png"
    
    fig, _ = plot_brier_vs_dilution_comparison(df, forecasters=forecasters,
                                               save_path=f"{output_dir}/{prefix}_iv_comparison.png")
    plt.close(fig)
    paths["iv_comparison"] = f"{output_dir}/{prefix}_iv_comparison.png"
    
    fig, _, _ = plot_regression_summary_table(df, forecasters=forecasters,
                                              save_path=f"{output_dir}/{prefix}_iv_table.png")
    plt.close(fig)
    paths["iv_table"] = f"{output_dir}/{prefix}_iv_table.png"
    
    return paths
