"""
CHRONOS Visualization Suite

Comprehensive visualization tools for CHRONOS strategy analysis,
including regime charts, performance plots, risk dashboards, and allocation views.

Main Entry Point:
    - generate_all_visualizations: Generate all plots from backtest results

Available Modules:
    - styles: Centralized styling and color palettes
    - regime_chart: Regime-colored price/portfolio charts
    - performance_plots: CHRONOS vs benchmark comparisons
    - transition_matrix: Regime transition probability heatmaps
    - risk_dashboard: Risk metrics visualization panels
    - allocation_plots: Portfolio allocation evolution charts
"""

from src.visualization.styles import (
    apply_chronos_style,
    save_figure,
    format_date_axis,
    REGIME_COLORS,
    REGIME_NAMES,
    ASSET_COLORS,
    CHRONOS_GREEN,
    BENCHMARK_BLUE
)

from src.visualization.regime_chart import (
    plot_regime_chart,
    plot_regime_overlay
)

from src.visualization.performance_plots import (
    plot_performance_comparison,
    plot_drawdown_comparison,
    plot_rolling_performance
)

from src.visualization.transition_matrix import (
    plot_transition_matrix,
    plot_regime_persistence,
    plot_transition_flow
)

from src.visualization.risk_dashboard import (
    plot_risk_metrics_panel,
    plot_rolling_sharpe
)

from src.visualization.allocation_plots import (
    plot_allocation_evolution,
    plot_regime_allocation_comparison,
    plot_allocation_changes,
    plot_allocation_summary
)

from src.visualization.generate_all_plots import (
    generate_all_visualizations
)

__all__ = [
    # Main entry point
    'generate_all_visualizations',
    
    # Styling utilities
    'apply_chronos_style',
    'save_figure',
    'format_date_axis',
    'REGIME_COLORS',
    'REGIME_NAMES',
    'ASSET_COLORS',
    'CHRONOS_GREEN',
    'BENCHMARK_BLUE',
    
    # Regime charts
    'plot_regime_chart',
    'plot_regime_overlay',
    
    # Performance plots
    'plot_performance_comparison',
    'plot_drawdown_comparison',
    'plot_rolling_performance',
    
    # Transition matrix
    'plot_transition_matrix',
    'plot_regime_persistence',
    'plot_transition_flow',
    
    # Risk dashboard
    'plot_risk_metrics_panel',
    'plot_rolling_sharpe',
    
    # Allocation plots
    'plot_allocation_evolution',
    'plot_regime_allocation_comparison',
    'plot_allocation_changes',
    'plot_allocation_summary',
]
