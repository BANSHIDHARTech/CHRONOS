"""
Master Visualization Generator

Orchestrates generation of all CHRONOS visualizations from backtest results.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import FIGURES_DIR, RESULTS_DIR, OUTPUTS_DIR
from src.visualization.styles import apply_chronos_style
from src.visualization.regime_chart import plot_regime_chart, plot_regime_overlay
from src.visualization.performance_plots import (
    plot_performance_comparison, plot_drawdown_comparison, plot_rolling_performance
)
from src.visualization.transition_matrix import (
    plot_transition_matrix, plot_regime_persistence, plot_transition_flow
)
from src.visualization.risk_dashboard import (
    plot_risk_metrics_panel, plot_rolling_sharpe
)
from src.visualization.allocation_plots import (
    plot_allocation_evolution, plot_regime_allocation_comparison,
    plot_allocation_changes, plot_allocation_summary
)
from src.utils.regime_utils import get_average_durations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_all_visualizations(
    results_df: pd.DataFrame,
    regime_detector=None,
    output_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate all CHRONOS visualizations from backtest results.
    
    Args:
        results_df: Backtest results DataFrame with required columns
        regime_detector: Optional RegimeDetector instance for transition matrix
        output_dir: Output directory (defaults to FIGURES_DIR)
        
    Returns:
        Dictionary mapping plot names to saved file paths
    """
    apply_chronos_style()
    
    if output_dir is None:
        output_dir = FIGURES_DIR
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'regime').mkdir(exist_ok=True)
    (output_path / 'performance').mkdir(exist_ok=True)
    (output_path / 'risk').mkdir(exist_ok=True)
    (output_path / 'allocation').mkdir(exist_ok=True)
    
    saved_files = {}
    
    # Ensure DatetimeIndex
    df = results_df.copy()
    if 'date' in df.columns:
        df = df.set_index('date')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # 1. Regime Chart
    try:
        if 'regime' in df.columns and 'portfolio_value' in df.columns:
            fig = plot_regime_overlay(
                df,
                price_column='portfolio_value',
                regime_column='regime',
                title="CHRONOS Portfolio Value with Market Regimes",
                save_path=str(output_path / 'regime' / 'regime_chart.png')
            )
            saved_files['regime_chart'] = str(output_path / 'regime' / 'regime_chart.png')
            logger.info("Generated regime_chart.png")
            fig.clf()
    except Exception as e:
        logger.error(f"Failed to generate regime chart: {e}")
    
    # 2. Performance Comparison
    try:
        if 'portfolio_value' in df.columns and 'benchmark_value' in df.columns:
            fig = plot_performance_comparison(
                df,
                save_path=str(output_path / 'performance' / 'performance_comparison.png')
            )
            saved_files['performance_comparison'] = str(output_path / 'performance' / 'performance_comparison.png')
            logger.info("Generated performance_comparison.png")
            fig.clf()
    except Exception as e:
        logger.error(f"Failed to generate performance comparison: {e}")
    
    # 3. Drawdown Comparison
    try:
        if 'portfolio_value' in df.columns and 'benchmark_value' in df.columns:
            fig1, fig2 = plot_drawdown_comparison(
                df,
                save_path=str(output_path / 'performance' / 'drawdown.png')
            )
            saved_files['drawdown_comparison'] = str(output_path / 'performance' / 'drawdown_drawdown.png')
            saved_files['drawdown_protection'] = str(output_path / 'performance' / 'drawdown_protection.png')
            logger.info("Generated drawdown comparison plots")
            fig1.clf()
            fig2.clf()
    except Exception as e:
        logger.error(f"Failed to generate drawdown comparison: {e}")
    
    # 4. Transition Matrix
    try:
        if regime_detector is not None:
            # Get transition matrix from regime detector
            trans_matrix = regime_detector.calculate_transition_probabilities()
            if isinstance(trans_matrix, pd.DataFrame):
                trans_matrix = trans_matrix.values
            
            fig = plot_transition_matrix(
                trans_matrix,
                save_path=str(output_path / 'regime' / 'transition_matrix.png')
            )
            saved_files['transition_matrix'] = str(output_path / 'regime' / 'transition_matrix.png')
            logger.info("Generated transition_matrix.png")
            fig.clf()
            
            # Transition flow diagram
            fig = plot_transition_flow(
                trans_matrix,
                save_path=str(output_path / 'regime' / 'transition_flow.png')
            )
            saved_files['transition_flow'] = str(output_path / 'regime' / 'transition_flow.png')
            logger.info("Generated transition_flow.png")
            fig.clf()
    except Exception as e:
        logger.error(f"Failed to generate transition matrix: {e}")
    
    # 5. Regime Persistence
    try:
        if 'regime' in df.columns:
            avg_durations = get_average_durations(df['regime'].values)
            fig = plot_regime_persistence(
                avg_durations,
                save_path=str(output_path / 'regime' / 'regime_persistence.png')
            )
            saved_files['regime_persistence'] = str(output_path / 'regime' / 'regime_persistence.png')
            logger.info("Generated regime_persistence.png")
            fig.clf()
    except Exception as e:
        logger.error(f"Failed to generate regime persistence: {e}")
    
    # 6. Risk Metrics Panel
    try:
        chronos_metrics, benchmark_metrics = _calculate_risk_metrics(df)
        fig = plot_risk_metrics_panel(
            chronos_metrics,
            benchmark_metrics,
            save_path=str(output_path / 'risk' / 'risk_metrics_panel.png')
        )
        saved_files['risk_metrics_panel'] = str(output_path / 'risk' / 'risk_metrics_panel.png')
        logger.info("Generated risk_metrics_panel.png")
        fig.clf()
    except Exception as e:
        logger.error(f"Failed to generate risk metrics panel: {e}")
    
    # 7. Rolling Sharpe
    try:
        if 'portfolio_value' in df.columns and 'benchmark_value' in df.columns:
            fig = plot_rolling_sharpe(
                df,
                window=26,  # 6-month window
                save_path=str(output_path / 'risk' / 'rolling_sharpe.png')
            )
            saved_files['rolling_sharpe'] = str(output_path / 'risk' / 'rolling_sharpe.png')
            logger.info("Generated rolling_sharpe.png")
            fig.clf()
    except Exception as e:
        logger.error(f"Failed to generate rolling Sharpe: {e}")
    
    # 8. Allocation Evolution
    try:
        weight_cols = [c for c in df.columns if c.endswith('_weight')]
        if weight_cols:
            fig = plot_allocation_evolution(
                df,
                weight_columns=weight_cols,
                save_path=str(output_path / 'allocation' / 'allocation_evolution.png')
            )
            saved_files['allocation_evolution'] = str(output_path / 'allocation' / 'allocation_evolution.png')
            logger.info("Generated allocation_evolution.png")
            fig.clf()
    except Exception as e:
        logger.error(f"Failed to generate allocation evolution: {e}")
    
    # 9. Regime Allocation Comparison
    try:
        weight_cols = [c for c in df.columns if c.endswith('_weight')]
        if weight_cols and 'regime' in df.columns:
            fig = plot_regime_allocation_comparison(
                df,
                weight_columns=weight_cols,
                save_path=str(output_path / 'allocation' / 'regime_allocation.png')
            )
            saved_files['regime_allocation'] = str(output_path / 'allocation' / 'regime_allocation.png')
            logger.info("Generated regime_allocation.png")
            fig.clf()
    except Exception as e:
        logger.error(f"Failed to generate regime allocation comparison: {e}")
    
    # 10. Allocation Changes
    try:
        weight_cols = [c for c in df.columns if c.endswith('_weight')]
        if weight_cols:
            fig = plot_allocation_changes(
                df,
                weight_columns=weight_cols,
                save_path=str(output_path / 'allocation' / 'allocation_changes.png')
            )
            saved_files['allocation_changes'] = str(output_path / 'allocation' / 'allocation_changes.png')
            logger.info("Generated allocation_changes.png")
            fig.clf()
    except Exception as e:
        logger.error(f"Failed to generate allocation changes: {e}")
    
    # 11. Allocation Summary
    try:
        weight_cols = [c for c in df.columns if c.endswith('_weight')]
        if weight_cols:
            fig = plot_allocation_summary(
                df,
                weight_columns=weight_cols,
                save_path=str(output_path / 'allocation' / 'allocation_summary.png')
            )
            saved_files['allocation_summary'] = str(output_path / 'allocation' / 'allocation_summary.png')
            logger.info("Generated allocation_summary.png")
            fig.clf()
    except Exception as e:
        logger.error(f"Failed to generate allocation summary: {e}")
    
    logger.info(f"Generated {len(saved_files)} visualizations to {output_dir}")
    
    return saved_files


def _calculate_risk_metrics(df: pd.DataFrame) -> tuple:
    """
    Calculate risk metrics for CHRONOS and benchmark from results DataFrame.
    
    Returns:
        Tuple of (chronos_metrics, benchmark_metrics) dictionaries
    """
    from src.portfolio.risk_metrics import RiskMetrics
    
    rm = RiskMetrics()
    
    # Calculate returns
    chronos_returns = df['portfolio_value'].pct_change().dropna()
    benchmark_returns = df['benchmark_value'].pct_change().dropna()
    
    # Get all metrics
    chronos_all = rm.calculate_all_metrics(chronos_returns)
    benchmark_all = rm.calculate_all_metrics(benchmark_returns)
    
    # Extract relevant metrics for visualization
    chronos_metrics = {
        'sharpe_ratio': chronos_all.get('sharpe_ratio', 0),
        'max_drawdown': chronos_all.get('max_drawdown', 0),
        'sortino_ratio': chronos_all.get('sortino_ratio', 0),
        'calmar_ratio': chronos_all.get('calmar_ratio', 0),
        'annual_return': chronos_all.get('annualized_return', 0),
        'annual_volatility': chronos_all.get('annualized_volatility', 0),
        'cvar_95': chronos_all.get('cvar_95', 0),
    }
    
    benchmark_metrics = {
        'sharpe_ratio': benchmark_all.get('sharpe_ratio', 0),
        'max_drawdown': benchmark_all.get('max_drawdown', 0),
        'sortino_ratio': benchmark_all.get('sortino_ratio', 0),
        'calmar_ratio': benchmark_all.get('calmar_ratio', 0),
        'annual_return': benchmark_all.get('annualized_return', 0),
        'annual_volatility': benchmark_all.get('annualized_volatility', 0),
        'cvar_95': benchmark_all.get('cvar_95', 0),
    }
    
    return chronos_metrics, benchmark_metrics


def main():
    """
    Main entry point for standalone visualization generation.
    
    Loads backtest results from default location and generates all plots.
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for file output
    
    # Load backtest results
    # Check multiple locations
    possible_paths = [
        Path(RESULTS_DIR) / 'backtest_results.csv',
        Path(OUTPUTS_DIR) / 'backtest' / 'backtest_results.csv'
    ]
    
    results_path = None
    for path in possible_paths:
        if path.exists():
            results_path = path
            break
            
    if results_path is None:
        logger.error(f"Backtest results not found. Checked locations:")
        for path in possible_paths:
            logger.error(f"  - {path}")
        logger.info("Please run the backtest first to generate results.")
        return
    
    logger.info(f"Loading backtest results from {results_path}")
    results_df = pd.read_csv(results_path, parse_dates=['date'])
    
    # Try to load regime detector
    regime_detector = None
    try:
        from src.models.regime_detector import RegimeDetector
        from config import MODELS_DIR
        
        model_path = Path(MODELS_DIR) / 'regime_detector.pkl'
        if model_path.exists():
            regime_detector = RegimeDetector.load_model(str(model_path))
            logger.info("Loaded regime detector model")
    except Exception as e:
        logger.warning(f"Could not load regime detector: {e}")
    
    # Generate all visualizations
    saved_files = generate_all_visualizations(
        results_df,
        regime_detector=regime_detector,
        output_dir=FIGURES_DIR
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("CHRONOS VISUALIZATION GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nGenerated {len(saved_files)} plots:")
    for name, path in saved_files.items():
        print(f"  - {name}: {path}")
    print(f"\nAll plots saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
