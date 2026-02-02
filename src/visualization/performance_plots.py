"""
Performance Comparison Plots Module

Creates CHRONOS vs benchmark comparison charts with crisis period highlighting.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.visualization.styles import (
    CHRONOS_GREEN, BENCHMARK_BLUE, CRISIS_COLOR, CRISIS_ALPHA,
    POSITIVE_COLOR, NEGATIVE_COLOR,
    save_figure, format_date_axis, apply_chronos_style,
    TITLE_FONTSIZE
)
from src.backtest.crisis_analyzer import CRISIS_PERIODS


def plot_performance_comparison(
    results_df: pd.DataFrame,
    portfolio_column: str = 'portfolio_value',
    benchmark_column: str = 'benchmark_value',
    show_crisis_periods: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create CHRONOS vs benchmark performance comparison chart.
    
    Shows cumulative returns with crisis period highlighting.
    
    Args:
        results_df: Backtest results DataFrame with DatetimeIndex
        portfolio_column: Column name for portfolio values
        benchmark_column: Column name for benchmark values
        show_crisis_periods: Whether to highlight crisis periods
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    apply_chronos_style()
    
    # Ensure DatetimeIndex
    df = results_df.copy()
    if 'date' in df.columns:
        df = df.set_index('date')
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Calculate cumulative returns as percentages
    initial_portfolio = df[portfolio_column].iloc[0]
    initial_benchmark = df[benchmark_column].iloc[0]
    
    chronos_returns = (df[portfolio_column] / initial_portfolio - 1) * 100
    benchmark_returns = (df[benchmark_column] / initial_benchmark - 1) * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot CHRONOS performance
    ax.plot(
        df.index, chronos_returns,
        color=CHRONOS_GREEN,
        linewidth=2.5,
        label="CHRONOS Strategy",
        zorder=3
    )
    
    # Plot benchmark performance
    ax.plot(
        df.index, benchmark_returns,
        color=BENCHMARK_BLUE,
        linewidth=2,
        alpha=0.7,
        label="S&P 500 Buy & Hold",
        zorder=3
    )
    
    # Add crisis period highlighting
    if show_crisis_periods:
        _add_crisis_highlights(ax, df.index)
    
    # Add zero line
    ax.axhline(y=0, linestyle='--', color='gray', alpha=0.5, zorder=2)
    
    # Formatting
    ax.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title(
        "CHRONOS vs S&P 500: Capital Preservation During Crises",
        fontsize=TITLE_FONTSIZE,
        fontweight='bold'
    )
    
    # Legend
    ax.legend(loc='upper left', framealpha=0.9)
    
    # Format x-axis
    format_date_axis(ax)
    
    # Grid
    ax.grid(True, alpha=0.3, zorder=1)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def _add_crisis_highlights(ax, dates: pd.DatetimeIndex):
    """
    Add vertical spans for crisis periods.
    
    Args:
        ax: Matplotlib axes object
        dates: DatetimeIndex of the data
    """
    date_min, date_max = dates.min(), dates.max()
    
    crisis_patches = []
    for crisis_id, crisis_info in CRISIS_PERIODS.items():
        start = pd.to_datetime(crisis_info['start'])
        end = pd.to_datetime(crisis_info['end'])
        
        # Skip if crisis is outside data range
        if end < date_min or start > date_max:
            continue
        
        # Clip to data range
        start = max(start, date_min)
        end = min(end, date_max)
        
        ax.axvspan(
            start, end,
            color=CRISIS_COLOR,
            alpha=CRISIS_ALPHA,
            zorder=1
        )
        
        # Add to legend only once
        if not crisis_patches:
            crisis_patches.append(mpatches.Patch(
                color=CRISIS_COLOR,
                alpha=CRISIS_ALPHA,
                label="Crisis Periods"
            ))
    
    # Add crisis patch to legend
    if crisis_patches:
        handles, labels = ax.get_legend_handles_labels()
        handles.extend(crisis_patches)
        ax.legend(handles=handles, loc='upper left', framealpha=0.9)


def plot_drawdown_comparison(
    results_df: pd.DataFrame,
    portfolio_column: str = 'portfolio_value',
    benchmark_column: str = 'benchmark_value',
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Create drawdown comparison charts.
    
    Creates two figures:
    1. Drawdown time series for both CHRONOS and benchmark
    2. Drawdown difference showing protection provided
    
    Args:
        results_df: Backtest results DataFrame with DatetimeIndex
        portfolio_column: Column name for portfolio values
        benchmark_column: Column name for benchmark values
        save_path: Optional base path for saving (appends _drawdown, _protection)
        
    Returns:
        Tuple of (drawdown_fig, protection_fig)
    """
    apply_chronos_style()
    
    # Ensure DatetimeIndex
    df = results_df.copy()
    if 'date' in df.columns:
        df = df.set_index('date')
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Calculate drawdowns
    chronos_dd = _calculate_drawdown(df[portfolio_column])
    benchmark_dd = _calculate_drawdown(df[benchmark_column])
    
    # Figure 1: Drawdown comparison
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    
    ax1.fill_between(
        df.index, chronos_dd * 100, 0,
        color=CHRONOS_GREEN,
        alpha=0.5,
        label="CHRONOS Drawdown"
    )
    ax1.fill_between(
        df.index, benchmark_dd * 100, 0,
        color=BENCHMARK_BLUE,
        alpha=0.5,
        label="S&P 500 Drawdown"
    )
    
    # Add max drawdown annotations
    chronos_max_dd = chronos_dd.min() * 100
    benchmark_max_dd = benchmark_dd.min() * 100
    chronos_max_idx = chronos_dd.idxmin()
    benchmark_max_idx = benchmark_dd.idxmin()
    
    ax1.annotate(
        f'Max: {chronos_max_dd:.1f}%',
        xy=(chronos_max_idx, chronos_max_dd),
        xytext=(10, -20),
        textcoords='offset points',
        fontsize=10,
        color=CHRONOS_GREEN,
        fontweight='bold'
    )
    
    ax1.annotate(
        f'Max: {benchmark_max_dd:.1f}%',
        xy=(benchmark_max_idx, benchmark_max_dd),
        xytext=(10, -20),
        textcoords='offset points',
        fontsize=10,
        color=BENCHMARK_BLUE,
        fontweight='bold'
    )
    
    ax1.set_ylabel("Drawdown (%)", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_title("Drawdown Comparison: CHRONOS vs S&P 500", fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax1.legend(loc='lower left', framealpha=0.9)
    format_date_axis(ax1)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Figure 2: Protection provided (difference)
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    
    protection = (benchmark_dd - chronos_dd) * 100  # Positive = CHRONOS protected better
    
    ax2.fill_between(
        df.index, protection, 0,
        where=(protection >= 0),
        color=POSITIVE_COLOR,
        alpha=0.6,
        label="CHRONOS Protection"
    )
    ax2.fill_between(
        df.index, protection, 0,
        where=(protection < 0),
        color=NEGATIVE_COLOR,
        alpha=0.6,
        label="Benchmark Advantage"
    )
    
    ax2.axhline(y=0, linestyle='-', color='black', alpha=0.3, linewidth=1)
    
    ax2.set_ylabel("Drawdown Protection (%)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_title("Downside Protection: CHRONOS vs S&P 500", fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax2.legend(loc='upper left', framealpha=0.9)
    format_date_axis(ax2)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        ext = save_path.rsplit('.', 1)[1] if '.' in save_path else 'png'
        save_figure(fig1, f"{base_path}_drawdown.{ext}")
        save_figure(fig2, f"{base_path}_protection.{ext}")
    
    return fig1, fig2


def _calculate_drawdown(values: pd.Series) -> pd.Series:
    """
    Calculate drawdown series from value series.
    
    Args:
        values: Series of portfolio values
        
    Returns:
        Series of drawdown values (negative percentages)
    """
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax
    return drawdown


def plot_rolling_performance(
    results_df: pd.DataFrame,
    window: int = 52,  # 52 weeks = 1 year
    portfolio_column: str = 'portfolio_value',
    benchmark_column: str = 'benchmark_value',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot rolling performance metrics.
    
    Args:
        results_df: Backtest results DataFrame
        window: Rolling window in weeks
        portfolio_column: Column name for portfolio values
        benchmark_column: Column name for benchmark values
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    apply_chronos_style()
    
    # Ensure DatetimeIndex
    df = results_df.copy()
    if 'date' in df.columns:
        df = df.set_index('date')
    
    # Calculate weekly returns
    chronos_returns = df[portfolio_column].pct_change()
    benchmark_returns = df[benchmark_column].pct_change()
    
    # Rolling return difference (alpha)
    alpha = chronos_returns.rolling(window=window).mean() - benchmark_returns.rolling(window=window).mean()
    alpha = alpha * 52 * 100  # Annualize and convert to percentage
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.fill_between(
        df.index, alpha, 0,
        where=(alpha >= 0),
        color=POSITIVE_COLOR,
        alpha=0.6,
        label="CHRONOS Outperformance"
    )
    ax.fill_between(
        df.index, alpha, 0,
        where=(alpha < 0),
        color=NEGATIVE_COLOR,
        alpha=0.6,
        label="Benchmark Outperformance"
    )
    
    ax.axhline(y=0, linestyle='-', color='black', alpha=0.5, linewidth=1)
    
    ax.set_ylabel("Rolling Alpha (% Annualized)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title(f"{window}-Week Rolling Alpha: CHRONOS vs S&P 500", fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    format_date_axis(ax)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


if __name__ == '__main__':
    # Test with synthetic data
    np.random.seed(42)
    n_points = 252
    
    dates = pd.date_range('2024-01-01', periods=n_points, freq='B')
    
    # Simulate CHRONOS outperforming during drawdowns
    benchmark_returns = np.random.randn(n_points) * 0.02
    chronos_returns = benchmark_returns * 0.7  # Less volatility
    chronos_returns[benchmark_returns < -0.02] *= 0.5  # Better during crashes
    
    benchmark_value = 100000 * np.cumprod(1 + benchmark_returns)
    portfolio_value = 100000 * np.cumprod(1 + chronos_returns)
    
    results_df = pd.DataFrame({
        'date': dates,
        'portfolio_value': portfolio_value,
        'benchmark_value': benchmark_value
    })
    
    fig1 = plot_performance_comparison(results_df)
    fig2, fig3 = plot_drawdown_comparison(results_df)
    plt.show()
