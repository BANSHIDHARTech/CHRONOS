"""
Allocation Evolution Plots Module

Visualizes portfolio weight changes and asset allocation over time.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from typing import Optional, List
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.visualization.styles import (
    ASSET_COLORS, REGIME_COLORS, REGIME_NAMES, REGIME_ALPHA,
    save_figure, format_date_axis, apply_chronos_style,
    TITLE_FONTSIZE
)
from config import PORTFOLIO_ASSETS


def plot_allocation_evolution(
    results_df: pd.DataFrame,
    weight_columns: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create stacked area chart showing portfolio allocation over time.
    
    Args:
        results_df: Backtest results DataFrame with weight columns
        weight_columns: List of weight column names (defaults to spy_weight, tlt_weight, gld_weight)
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
    
    # Default weight columns
    if weight_columns is None:
        weight_columns = ['spy_weight', 'tlt_weight', 'gld_weight']
    
    # Extract weights (convert to percentages)
    weights_data = []
    labels = []
    colors = []
    
    for col in weight_columns:
        if col in df.columns:
            weights_data.append(df[col].values * 100)
            
            # Extract asset name from column
            asset = col.replace('_weight', '').upper()
            labels.append(asset)
            colors.append(ASSET_COLORS.get(asset, '#808080'))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create stacked area chart
    ax.stackplot(
        df.index,
        weights_data,
        labels=labels,
        colors=colors,
        alpha=0.7,
        edgecolor='white',
        linewidth=0.5
    )
    
    # Add reference lines
    for pct in [25, 50, 75]:
        ax.axhline(y=pct, linestyle='--', color='gray', alpha=0.3, linewidth=1)
    
    # Formatting
    ax.set_ylabel("Portfolio Weight (%)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title("Portfolio Allocation Over Time", fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.set_ylim(0, 100)
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Format x-axis
    format_date_axis(ax)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_regime_allocation_comparison(
    results_df: pd.DataFrame,
    weight_columns: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create grouped bar chart showing average allocation by regime.
    
    Args:
        results_df: Backtest results DataFrame with weight and regime columns
        weight_columns: List of weight column names
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    apply_chronos_style()
    
    # Ensure DatetimeIndex
    df = results_df.copy()
    if 'date' in df.columns:
        df = df.set_index('date')
    
    # Default weight columns
    if weight_columns is None:
        weight_columns = ['spy_weight', 'tlt_weight', 'gld_weight']
    
    # Calculate average weights per regime
    avg_weights = {}
    for regime_id in range(3):
        regime_data = df[df['regime'] == regime_id]
        if len(regime_data) > 0:
            avg_weights[regime_id] = {
                col.replace('_weight', '').upper(): regime_data[col].mean() * 100
                for col in weight_columns if col in df.columns
            }
    
    # Prepare data for plotting
    regimes = list(avg_weights.keys())
    regime_labels = [REGIME_NAMES[r] for r in regimes]
    assets = list(avg_weights[regimes[0]].keys()) if regimes else []
    
    x = np.arange(len(regimes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot bars for each asset
    for i, asset in enumerate(assets):
        values = [avg_weights[r].get(asset, 0) for r in regimes]
        offset = (i - len(assets) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width,
            color=ASSET_COLORS.get(asset, '#808080'),
            label=asset,
            edgecolor='white',
            linewidth=1
        )
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{val:.0f}%',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold'
            )
    
    # Color-code regime labels
    ax.set_xticks(x)
    ax.set_xticklabels(regime_labels, fontsize=11)
    
    # Add colored patches behind x-tick labels
    for i, (regime_id, label) in enumerate(zip(regimes, regime_labels)):
        ax.get_xticklabels()[i].set_color(REGIME_COLORS[regime_id])
        ax.get_xticklabels()[i].set_fontweight('bold')
    
    ax.set_ylabel("Average Weight (%)", fontsize=12)
    ax.set_title("Average Asset Allocation by Market Regime", fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_allocation_changes(
    results_df: pd.DataFrame,
    weight_columns: Optional[List[str]] = None,
    change_threshold: float = 0.10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot week-over-week weight changes with rebalancing highlights.
    
    Args:
        results_df: Backtest results DataFrame with weight columns
        weight_columns: List of weight column names
        change_threshold: Threshold for highlighting large rebalances (e.g., 0.10 = 10%)
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
    
    # Default weight columns
    if weight_columns is None:
        weight_columns = ['spy_weight', 'tlt_weight', 'gld_weight']
    
    # Calculate weight changes
    fig, axes = plt.subplots(len(weight_columns), 1, figsize=(14, 3 * len(weight_columns)), sharex=True)
    
    if len(weight_columns) == 1:
        axes = [axes]
    
    for ax, col in zip(axes, weight_columns):
        if col not in df.columns:
            continue
        
        asset = col.replace('_weight', '').upper()
        weight_change = df[col].diff() * 100  # Convert to percentage change
        
        # Plot weight changes
        ax.fill_between(
            df.index, weight_change, 0,
            where=(weight_change >= 0),
            color=ASSET_COLORS.get(asset, '#808080'),
            alpha=0.6,
            label=f'{asset} Increase'
        )
        ax.fill_between(
            df.index, weight_change, 0,
            where=(weight_change < 0),
            color=ASSET_COLORS.get(asset, '#808080'),
            alpha=0.3,
            label=f'{asset} Decrease'
        )
        
        # Highlight large rebalancing events
        large_changes = df[abs(df[col].diff()) > change_threshold].index
        for date in large_changes:
            ax.axvline(x=date, color='red', alpha=0.3, linewidth=1)
        
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
        ax.set_ylabel(f'{asset} Change (%)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    # Add regime backgrounds to bottom plot if regime column exists
    if 'regime' in df.columns:
        _add_regime_backgrounds_to_ax(axes[-1], df)
    
    # Formatting
    axes[-1].set_xlabel("Date", fontsize=12)
    format_date_axis(axes[-1])
    
    fig.suptitle(
        "Portfolio Weight Changes Over Time",
        fontsize=TITLE_FONTSIZE, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def _add_regime_backgrounds_to_ax(ax, df: pd.DataFrame):
    """Add regime-colored backgrounds to an axis."""
    if 'regime' not in df.columns:
        return
    
    regimes = df['regime'].values
    dates = df.index
    
    if len(dates) == 0:
        return
    
    current_regime = regimes[0]
    start_idx = 0
    
    for i in range(1, len(regimes)):
        if regimes[i] != current_regime:
            ax.axvspan(
                dates[start_idx], dates[i - 1],
                color=REGIME_COLORS.get(current_regime, '#808080'),
                alpha=0.1,
                zorder=0
            )
            current_regime = regimes[i]
            start_idx = i
    
    # Final regime period
    ax.axvspan(
        dates[start_idx], dates[-1],
        color=REGIME_COLORS.get(current_regime, '#808080'),
        alpha=0.1,
        zorder=0
    )


def plot_allocation_summary(
    results_df: pd.DataFrame,
    weight_columns: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a combined allocation summary with pie charts and evolution.
    
    Args:
        results_df: Backtest results DataFrame with weight columns
        weight_columns: List of weight column names
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    apply_chronos_style()
    
    # Ensure DatetimeIndex
    df = results_df.copy()
    if 'date' in df.columns:
        df = df.set_index('date')
    
    # Default weight columns
    if weight_columns is None:
        weight_columns = ['spy_weight', 'tlt_weight', 'gld_weight']
    
    fig = plt.figure(figsize=(16, 8))
    
    # Subplot 1: Evolution (larger)
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    
    # Plot stacked area
    weights_data = []
    labels = []
    colors = []
    
    for col in weight_columns:
        if col in df.columns:
            weights_data.append(df[col].values * 100)
            asset = col.replace('_weight', '').upper()
            labels.append(asset)
            colors.append(ASSET_COLORS.get(asset, '#808080'))
    
    ax1.stackplot(
        df.index, weights_data,
        labels=labels, colors=colors,
        alpha=0.7, edgecolor='white', linewidth=0.5
    )
    
    ax1.set_ylabel("Portfolio Weight (%)", fontsize=12)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_title("Portfolio Allocation Evolution", fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper left', framealpha=0.9)
    format_date_axis(ax1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: Average allocation pie chart
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    
    avg_weights = [df[col].mean() * 100 for col in weight_columns if col in df.columns]
    ax2.pie(
        avg_weights, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    ax2.set_title("Average Allocation", fontsize=12, fontweight='bold')
    
    # Subplot 3: Latest allocation pie chart
    ax3 = plt.subplot2grid((2, 3), (1, 2))
    
    latest_weights = [df[col].iloc[-1] * 100 for col in weight_columns if col in df.columns]
    ax3.pie(
        latest_weights, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    ax3.set_title("Current Allocation", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


if __name__ == '__main__':
    # Test with synthetic data
    np.random.seed(42)
    n_points = 252
    
    dates = pd.date_range('2024-01-01', periods=n_points, freq='B')
    
    # Simulate regime-dependent allocations
    regimes = np.random.choice([0, 1, 2], size=n_points, p=[0.4, 0.35, 0.25])
    
    spy_weight = np.where(regimes == 0, 0.7, np.where(regimes == 1, 0.5, 0.1))
    spy_weight += np.random.randn(n_points) * 0.05
    spy_weight = np.clip(spy_weight, 0, 1)
    
    tlt_weight = np.where(regimes == 0, 0.2, np.where(regimes == 1, 0.3, 0.6))
    tlt_weight += np.random.randn(n_points) * 0.03
    tlt_weight = np.clip(tlt_weight, 0, 1)
    
    gld_weight = 1 - spy_weight - tlt_weight
    gld_weight = np.clip(gld_weight, 0, 1)
    
    # Normalize
    total = spy_weight + tlt_weight + gld_weight
    spy_weight /= total
    tlt_weight /= total
    gld_weight /= total
    
    results_df = pd.DataFrame({
        'date': dates,
        'spy_weight': spy_weight,
        'tlt_weight': tlt_weight,
        'gld_weight': gld_weight,
        'regime': regimes
    })
    
    fig1 = plot_allocation_evolution(results_df)
    fig2 = plot_regime_allocation_comparison(results_df)
    fig3 = plot_allocation_changes(results_df)
    fig4 = plot_allocation_summary(results_df)
    
    plt.show()
