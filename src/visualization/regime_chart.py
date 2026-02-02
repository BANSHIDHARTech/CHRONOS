"""
Regime-Colored Price Chart Module

Creates the signature CHRONOS visualization showing price movement
with regime-colored backgrounds.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from typing import Optional, Union
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.visualization.styles import (
    REGIME_COLORS, REGIME_NAMES, REGIME_ALPHA,
    save_figure, format_date_axis, apply_chronos_style,
    TITLE_FONTSIZE
)


def plot_regime_chart(
    prices: Union[pd.Series, np.ndarray],
    regimes: Union[pd.Series, np.ndarray],
    dates: pd.DatetimeIndex,
    title: str = "CHRONOS Regime-Based Market Analysis",
    ticker_name: str = "S&P 500",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a price chart with regime-colored background spans.
    
    This is the signature CHRONOS visualization showing how regimes
    align with price movements.
    
    Args:
        prices: Price series data
        regimes: Regime labels (0, 1, 2) aligned with prices
        dates: DatetimeIndex aligned with prices
        title: Chart title
        ticker_name: Name for the price series in legend
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    apply_chronos_style()
    
    # Convert to numpy arrays if needed
    if isinstance(prices, pd.Series):
        prices = prices.values
    if isinstance(regimes, pd.Series):
        regimes = regimes.values
    
    # Validate lengths
    if len(prices) != len(regimes) or len(prices) != len(dates):
        raise ValueError("prices, regimes, and dates must have the same length")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot price line
    ax.plot(dates, prices, color='black', linewidth=2, label=ticker_name, zorder=3)
    
    # Add regime-colored backgrounds using axvspan
    _add_regime_backgrounds(ax, dates, regimes)
    
    # Create custom legend for regimes
    legend_patches = []
    for regime_id, regime_name in REGIME_NAMES.items():
        patch = mpatches.Patch(
            color=REGIME_COLORS[regime_id],
            alpha=REGIME_ALPHA,
            label=regime_name
        )
        legend_patches.append(patch)
    
    # Add price legend element
    from matplotlib.lines import Line2D
    price_line = Line2D([0], [0], color='black', linewidth=2, label=ticker_name)
    legend_patches.insert(0, price_line)
    
    # Add legend
    ax.legend(handles=legend_patches, loc='upper left', framealpha=0.9)
    
    # Formatting
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold')
    
    # Format x-axis
    format_date_axis(ax)
    
    # Grid
    ax.grid(True, alpha=0.3, zorder=1)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def _add_regime_backgrounds(ax, dates: pd.DatetimeIndex, regimes: np.ndarray):
    """
    Add colored background spans for each regime period.
    
    Args:
        ax: Matplotlib axes object
        dates: DatetimeIndex
        regimes: Array of regime labels
    """
    if len(dates) == 0:
        return
    
    current_regime = regimes[0]
    start_idx = 0
    
    for i in range(1, len(regimes)):
        if regimes[i] != current_regime:
            # Draw span for completed regime period
            ax.axvspan(
                dates[start_idx],
                dates[i - 1],
                color=REGIME_COLORS.get(current_regime, '#808080'),
                alpha=REGIME_ALPHA,
                zorder=1
            )
            current_regime = regimes[i]
            start_idx = i
    
    # Draw final regime period
    ax.axvspan(
        dates[start_idx],
        dates[-1],
        color=REGIME_COLORS.get(current_regime, '#808080'),
        alpha=REGIME_ALPHA,
        zorder=1
    )


def plot_regime_overlay(
    results_df: pd.DataFrame,
    price_column: str = 'portfolio_value',
    regime_column: str = 'regime',
    title: str = "Portfolio Value with Market Regimes",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot portfolio value with regime overlays from backtest results.
    
    Convenience function that extracts data from results DataFrame.
    
    Args:
        results_df: Backtest results DataFrame with DatetimeIndex
        price_column: Column name for price/value data
        regime_column: Column name for regime labels
        title: Chart title
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    # Ensure DatetimeIndex
    if 'date' in results_df.columns:
        results_df = results_df.set_index('date')
    
    dates = results_df.index
    prices = results_df[price_column].values
    regimes = results_df[regime_column].values
    
    return plot_regime_chart(
        prices=prices,
        regimes=regimes,
        dates=dates,
        title=title,
        ticker_name="Portfolio Value",
        save_path=save_path
    )


if __name__ == '__main__':
    # Test with synthetic data
    import numpy as np
    
    np.random.seed(42)
    n_points = 252  # One year of trading days
    
    dates = pd.date_range('2024-01-01', periods=n_points, freq='B')
    prices = 100 * np.cumprod(1 + np.random.randn(n_points) * 0.01)
    regimes = np.random.choice([0, 1, 2], size=n_points, p=[0.4, 0.35, 0.25])
    
    fig = plot_regime_chart(prices, regimes, dates, title="Test Regime Chart")
    plt.show()
