"""
Risk Metrics Dashboard Module

Creates comprehensive risk metrics visualization panels comparing
CHRONOS strategy against benchmarks.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.visualization.styles import (
    CHRONOS_GREEN, BENCHMARK_BLUE, POSITIVE_COLOR, NEGATIVE_COLOR,
    save_figure, format_date_axis, apply_chronos_style,
    TITLE_FONTSIZE
)


def plot_risk_metrics_panel(
    chronos_metrics: Dict[str, float],
    benchmark_metrics: Dict[str, float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive risk metrics dashboard panel.
    
    Shows 6 subplots comparing various risk metrics between CHRONOS and benchmark.
    
    Args:
        chronos_metrics: Dictionary with CHRONOS risk metrics
        benchmark_metrics: Dictionary with benchmark risk metrics
        save_path: Optional path to save the figure
        
    Expected metrics keys:
        - sharpe_ratio
        - max_drawdown
        - sortino_ratio
        - calmar_ratio
        - annual_return
        - annual_volatility
        - cvar_95
        
    Returns:
        Matplotlib Figure object
    """
    apply_chronos_style()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    # Subplot 1: Sharpe Ratio Comparison
    _plot_bar_comparison(
        axes[0],
        'Sharpe Ratio',
        chronos_metrics.get('sharpe_ratio', 0),
        benchmark_metrics.get('sharpe_ratio', 0),
        format_str='.2f'
    )
    
    # Subplot 2: Max Drawdown Comparison (horizontal bars, negative values)
    _plot_drawdown_bars(
        axes[1],
        chronos_metrics.get('max_drawdown', 0),
        benchmark_metrics.get('max_drawdown', 0)
    )
    
    # Subplot 3: Sortino and Calmar Ratios (grouped bar chart)
    _plot_grouped_ratios(
        axes[2],
        chronos_metrics,
        benchmark_metrics
    )
    
    # Subplot 4: Return vs Volatility Scatter
    _plot_return_volatility(
        axes[3],
        chronos_metrics,
        benchmark_metrics
    )
    
    # Subplot 5: CVaR Comparison
    _plot_cvar_comparison(
        axes[4],
        chronos_metrics.get('cvar_95', 0),
        benchmark_metrics.get('cvar_95', 0)
    )
    
    # Subplot 6: Summary Table
    _plot_summary_table(
        axes[5],
        chronos_metrics,
        benchmark_metrics
    )
    
    # Overall title
    fig.suptitle(
        "CHRONOS Risk-Adjusted Performance Metrics",
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def _plot_bar_comparison(ax, title: str, chronos_val: float, benchmark_val: float, format_str: str = '.2f'):
    """Helper to create simple bar comparison."""
    x = np.arange(2)
    values = [chronos_val, benchmark_val]
    colors = [CHRONOS_GREEN, BENCHMARK_BLUE]
    labels = ['CHRONOS', 'S&P 500']
    
    bars = ax.bar(x, values, color=colors, edgecolor='white', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02 * max(abs(v) for v in values),
            f'{val:{format_str}}',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold'
        )
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add good/bad threshold line for Sharpe
    if 'Sharpe' in title:
        ax.axhline(y=1.0, linestyle='--', color='gray', alpha=0.5, label='Good (1.0)')
        ax.legend(loc='upper right', fontsize=9)


def _plot_drawdown_bars(ax, chronos_dd: float, benchmark_dd: float):
    """Helper to create horizontal drawdown bars."""
    y = np.arange(2)
    # Drawdowns are negative, so we plot absolute values
    values = [abs(chronos_dd) * 100, abs(benchmark_dd) * 100]
    colors = [CHRONOS_GREEN, BENCHMARK_BLUE]
    labels = ['CHRONOS', 'S&P 500']
    
    bars = ax.barh(y, values, color=colors, edgecolor='white', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'-{val:.1f}%',
            va='center',
            fontsize=11, fontweight='bold'
        )
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Maximum Drawdown (%)")
    ax.set_title("Max Drawdown (Lower is Better)", fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(values) * 1.3)
    ax.grid(True, alpha=0.3, axis='x')


def _plot_grouped_ratios(ax, chronos_metrics: Dict, benchmark_metrics: Dict):
    """Helper to create grouped bar chart for ratios."""
    metrics = ['sortino_ratio', 'calmar_ratio']
    metric_labels = ['Sortino Ratio', 'Calmar Ratio']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    chronos_vals = [chronos_metrics.get(m, 0) for m in metrics]
    benchmark_vals = [benchmark_metrics.get(m, 0) for m in metrics]
    
    bars1 = ax.bar(x - width/2, chronos_vals, width, color=CHRONOS_GREEN, label='CHRONOS')
    bars2 = ax.bar(x + width/2, benchmark_vals, width, color=BENCHMARK_BLUE, label='S&P 500')
    
    # Add value labels
    for bars, vals in [(bars1, chronos_vals), (bars2, benchmark_vals)]:
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f'{val:.2f}',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold'
            )
    
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_title("Risk-Adjusted Ratios", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')


def _plot_return_volatility(ax, chronos_metrics: Dict, benchmark_metrics: Dict):
    """Helper to create return vs volatility scatter."""
    chronos_ret = chronos_metrics.get('annual_return', 0) * 100
    chronos_vol = chronos_metrics.get('annual_volatility', 0) * 100
    benchmark_ret = benchmark_metrics.get('annual_return', 0) * 100
    benchmark_vol = benchmark_metrics.get('annual_volatility', 0) * 100
    
    ax.scatter(
        chronos_vol, chronos_ret,
        s=200, c=CHRONOS_GREEN, marker='o',
        edgecolors='white', linewidth=2,
        label='CHRONOS', zorder=3
    )
    ax.scatter(
        benchmark_vol, benchmark_ret,
        s=200, c=BENCHMARK_BLUE, marker='s',
        edgecolors='white', linewidth=2,
        label='S&P 500', zorder=3
    )
    
    # Add labels
    ax.annotate(
        'CHRONOS',
        xy=(chronos_vol, chronos_ret),
        xytext=(10, 10), textcoords='offset points',
        fontsize=10, fontweight='bold', color=CHRONOS_GREEN
    )
    ax.annotate(
        'S&P 500',
        xy=(benchmark_vol, benchmark_ret),
        xytext=(10, -15), textcoords='offset points',
        fontsize=10, fontweight='bold', color=BENCHMARK_BLUE
    )
    
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title("Return vs Volatility", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)


def _plot_cvar_comparison(ax, chronos_cvar: float, benchmark_cvar: float):
    """Helper to create CVaR comparison bar chart."""
    x = np.arange(2)
    # CVaR is typically negative or expressed as loss
    values = [abs(chronos_cvar) * 100, abs(benchmark_cvar) * 100]
    colors = [CHRONOS_GREEN, BENCHMARK_BLUE]
    labels = ['CHRONOS', 'S&P 500']
    
    bars = ax.bar(x, values, color=colors, edgecolor='white', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f'{val:.2f}%',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold'
        )
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CVaR 95% (Expected Tail Loss %)")
    ax.set_title("Conditional Value at Risk (Lower is Better)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')


def _plot_summary_table(ax, chronos_metrics: Dict, benchmark_metrics: Dict):
    """Helper to create summary metrics table."""
    ax.axis('off')
    
    # Prepare table data
    metrics_list = [
        ('Annual Return', 'annual_return', '{:.1%}'),
        ('Annual Volatility', 'annual_volatility', '{:.1%}'),
        ('Sharpe Ratio', 'sharpe_ratio', '{:.2f}'),
        ('Sortino Ratio', 'sortino_ratio', '{:.2f}'),
        ('Max Drawdown', 'max_drawdown', '{:.1%}'),
        ('Calmar Ratio', 'calmar_ratio', '{:.2f}'),
    ]
    
    table_data = []
    for label, key, fmt in metrics_list:
        chronos_val = chronos_metrics.get(key, 0)
        benchmark_val = benchmark_metrics.get(key, 0)
        table_data.append([
            label,
            fmt.format(chronos_val),
            fmt.format(benchmark_val)
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Metric', 'CHRONOS', 'S&P 500'],
        cellLoc='center',
        loc='center',
        colWidths=[0.4, 0.3, 0.3]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#E0E0E0')
        table[(0, i)].set_text_props(fontweight='bold')
    
    # Color CHRONOS column green, benchmark blue
    for i in range(1, len(table_data) + 1):
        table[(i, 1)].set_facecolor('#E8F5E9')  # Light green
        table[(i, 2)].set_facecolor('#E3F2FD')  # Light blue
    
    ax.set_title("Performance Summary", fontsize=12, fontweight='bold', pad=20)


def plot_rolling_sharpe(
    results_df: pd.DataFrame,
    window: int = 52,
    portfolio_column: str = 'portfolio_value',
    benchmark_column: str = 'benchmark_value',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot rolling Sharpe ratio comparison.
    
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
    
    # Calculate rolling Sharpe (annualized)
    def rolling_sharpe(returns, window):
        rolling_mean = returns.rolling(window=window).mean() * 52
        rolling_std = returns.rolling(window=window).std() * np.sqrt(52)
        return rolling_mean / rolling_std
    
    chronos_sharpe = rolling_sharpe(chronos_returns, window)
    benchmark_sharpe = rolling_sharpe(benchmark_returns, window)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(
        df.index, chronos_sharpe,
        color=CHRONOS_GREEN, linewidth=2,
        label='CHRONOS'
    )
    ax.plot(
        df.index, benchmark_sharpe,
        color=BENCHMARK_BLUE, linewidth=2, alpha=0.7,
        label='S&P 500'
    )
    
    # Add threshold line at Sharpe = 1
    ax.axhline(y=1.0, linestyle='--', color='green', alpha=0.5, label='Good (1.0)')
    ax.axhline(y=0, linestyle='-', color='gray', alpha=0.3)
    
    # Confidence band (simplified - standard error)
    chronos_se = chronos_sharpe.rolling(window=window//2).std() / np.sqrt(window)
    ax.fill_between(
        df.index,
        chronos_sharpe - 1.96 * chronos_se,
        chronos_sharpe + 1.96 * chronos_se,
        color=CHRONOS_GREEN, alpha=0.1
    )
    
    ax.set_ylabel("Rolling Sharpe Ratio", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title(f"{window}-Week Rolling Sharpe Ratio", fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    format_date_axis(ax)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


if __name__ == '__main__':
    # Test with sample metrics
    chronos_metrics = {
        'sharpe_ratio': 1.45,
        'max_drawdown': -0.12,
        'sortino_ratio': 2.10,
        'calmar_ratio': 1.25,
        'annual_return': 0.15,
        'annual_volatility': 0.10,
        'cvar_95': -0.025
    }
    
    benchmark_metrics = {
        'sharpe_ratio': 0.85,
        'max_drawdown': -0.25,
        'sortino_ratio': 1.20,
        'calmar_ratio': 0.65,
        'annual_return': 0.12,
        'annual_volatility': 0.18,
        'cvar_95': -0.045
    }
    
    fig = plot_risk_metrics_panel(chronos_metrics, benchmark_metrics)
    plt.show()
