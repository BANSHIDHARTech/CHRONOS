"""
Visualization Module Integration Tests

Tests for all CHRONOS visualization functions to ensure correct
figure generation and output.
"""

import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.styles import (
    REGIME_COLORS, REGIME_NAMES, apply_chronos_style, save_figure
)
from src.visualization.regime_chart import plot_regime_chart, plot_regime_overlay
from src.visualization.performance_plots import (
    plot_performance_comparison, plot_drawdown_comparison
)
from src.visualization.transition_matrix import (
    plot_transition_matrix, plot_regime_persistence
)
from src.visualization.risk_dashboard import plot_risk_metrics_panel, plot_rolling_sharpe
from src.visualization.allocation_plots import (
    plot_allocation_evolution, plot_regime_allocation_comparison,
    plot_allocation_changes
)
from src.visualization.generate_all_plots import generate_all_visualizations


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_backtest_results():
    """Create synthetic backtest results DataFrame for testing."""
    np.random.seed(42)
    n_points = 252
    
    dates = pd.date_range('2024-01-01', periods=n_points, freq='B')
    
    # Simulate portfolio and benchmark values
    benchmark_returns = np.random.randn(n_points) * 0.015
    chronos_returns = benchmark_returns * 0.7
    chronos_returns[benchmark_returns < -0.02] *= 0.5
    
    portfolio_value = 100000 * np.cumprod(1 + chronos_returns)
    benchmark_value = 100000 * np.cumprod(1 + benchmark_returns)
    
    # Simulate regimes
    regimes = np.random.choice([0, 1, 2], size=n_points, p=[0.4, 0.35, 0.25])
    
    # Simulate weights
    spy_weight = np.where(regimes == 0, 0.7, np.where(regimes == 1, 0.5, 0.15))
    spy_weight += np.random.randn(n_points) * 0.03
    spy_weight = np.clip(spy_weight, 0, 1)
    
    tlt_weight = np.where(regimes == 0, 0.2, np.where(regimes == 1, 0.3, 0.55))
    tlt_weight += np.random.randn(n_points) * 0.02
    tlt_weight = np.clip(tlt_weight, 0, 1)
    
    gld_weight = 1 - spy_weight - tlt_weight
    gld_weight = np.clip(gld_weight, 0, 1)
    
    # Normalize
    total = spy_weight + tlt_weight + gld_weight
    spy_weight /= total
    tlt_weight /= total
    gld_weight /= total
    
    df = pd.DataFrame({
        'date': dates,
        'portfolio_value': portfolio_value,
        'benchmark_value': benchmark_value,
        'regime': regimes,
        'spy_weight': spy_weight,
        'tlt_weight': tlt_weight,
        'gld_weight': gld_weight,
        'weekly_return': np.random.randn(n_points) * 0.02,
        'drawdown': np.minimum(0, np.random.randn(n_points) * 0.05)
    })
    
    return df


@pytest.fixture
def sample_transition_matrix():
    """Create sample 3x3 transition probability matrix."""
    return np.array([
        [0.85, 0.10, 0.05],
        [0.15, 0.75, 0.10],
        [0.05, 0.20, 0.75],
    ])


@pytest.fixture
def sample_risk_metrics():
    """Create sample risk metrics dictionaries."""
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
    
    return chronos_metrics, benchmark_metrics


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# STYLE TESTS
# ============================================================================

class TestStyles:
    """Tests for visualization styles module."""
    
    def test_regime_colors_defined(self):
        """Verify all regime colors are defined."""
        assert 0 in REGIME_COLORS
        assert 1 in REGIME_COLORS
        assert 2 in REGIME_COLORS
        assert all(c.startswith('#') for c in REGIME_COLORS.values())
    
    def test_regime_names_defined(self):
        """Verify all regime names are defined."""
        assert REGIME_NAMES[0] == 'Euphoria'
        assert REGIME_NAMES[1] == 'Complacency'
        assert REGIME_NAMES[2] == 'Capitulation'
    
    def test_apply_chronos_style(self):
        """Verify style application doesn't error."""
        apply_chronos_style()
        assert plt.rcParams['figure.dpi'] == 150
    
    def test_save_figure(self, temp_output_dir):
        """Verify figure saving works correctly."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        filepath = save_figure(fig, 'test_figure.png', temp_output_dir)
        
        assert Path(filepath).exists()
        assert filepath.endswith('test_figure.png')
        
        plt.close(fig)


# ============================================================================
# REGIME CHART TESTS
# ============================================================================

class TestRegimeChart:
    """Tests for regime chart visualization."""
    
    def test_plot_regime_chart_returns_figure(self, sample_backtest_results):
        """Verify plot_regime_chart returns a Figure object."""
        df = sample_backtest_results.set_index('date')
        
        fig = plot_regime_chart(
            prices=df['portfolio_value'].values,
            regimes=df['regime'].values,
            dates=df.index
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_regime_chart_saves_file(self, sample_backtest_results, temp_output_dir):
        """Verify plot_regime_chart saves to file when path provided."""
        df = sample_backtest_results.set_index('date')
        save_path = os.path.join(temp_output_dir, 'regime_test.png')
        
        fig = plot_regime_chart(
            prices=df['portfolio_value'].values,
            regimes=df['regime'].values,
            dates=df.index,
            save_path=save_path
        )
        
        assert Path(save_path).exists()
        plt.close(fig)
    
    def test_plot_regime_overlay(self, sample_backtest_results):
        """Verify plot_regime_overlay works with DataFrame."""
        fig = plot_regime_overlay(sample_backtest_results)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ============================================================================
# PERFORMANCE PLOTS TESTS
# ============================================================================

class TestPerformancePlots:
    """Tests for performance comparison plots."""
    
    def test_plot_performance_comparison_returns_figure(self, sample_backtest_results):
        """Verify plot_performance_comparison returns a Figure."""
        fig = plot_performance_comparison(sample_backtest_results)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_performance_comparison_saves_file(self, sample_backtest_results, temp_output_dir):
        """Verify performance comparison saves to file."""
        save_path = os.path.join(temp_output_dir, 'perf_test.png')
        
        fig = plot_performance_comparison(
            sample_backtest_results,
            save_path=save_path
        )
        
        assert Path(save_path).exists()
        plt.close(fig)
    
    def test_plot_drawdown_comparison_returns_two_figures(self, sample_backtest_results):
        """Verify drawdown comparison returns two figures."""
        fig1, fig2 = plot_drawdown_comparison(sample_backtest_results)
        
        assert isinstance(fig1, plt.Figure)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig1)
        plt.close(fig2)


# ============================================================================
# TRANSITION MATRIX TESTS
# ============================================================================

class TestTransitionMatrix:
    """Tests for transition matrix visualization."""
    
    def test_plot_transition_matrix_numpy(self, sample_transition_matrix):
        """Verify transition matrix plot works with numpy array."""
        fig = plot_transition_matrix(sample_transition_matrix)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_transition_matrix_dataframe(self, sample_transition_matrix):
        """Verify transition matrix plot works with DataFrame."""
        df = pd.DataFrame(
            sample_transition_matrix,
            index=['Euphoria', 'Complacency', 'Capitulation'],
            columns=['Euphoria', 'Complacency', 'Capitulation']
        )
        
        fig = plot_transition_matrix(df)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_regime_persistence(self):
        """Verify regime persistence bar chart works."""
        avg_durations = {0: 15.5, 1: 22.3, 2: 8.7}
        
        fig = plot_regime_persistence(avg_durations)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ============================================================================
# RISK DASHBOARD TESTS
# ============================================================================

class TestRiskDashboard:
    """Tests for risk dashboard visualization."""
    
    def test_plot_risk_metrics_panel(self, sample_risk_metrics):
        """Verify risk metrics panel creates all subplots."""
        chronos_metrics, benchmark_metrics = sample_risk_metrics
        
        fig = plot_risk_metrics_panel(chronos_metrics, benchmark_metrics)
        
        assert isinstance(fig, plt.Figure)
        # Should have 6 subplots in 2x3 grid
        assert len(fig.axes) >= 6
        plt.close(fig)
    
    def test_plot_risk_metrics_panel_saves_file(self, sample_risk_metrics, temp_output_dir):
        """Verify risk metrics panel saves to file."""
        chronos_metrics, benchmark_metrics = sample_risk_metrics
        save_path = os.path.join(temp_output_dir, 'risk_panel.png')
        
        fig = plot_risk_metrics_panel(
            chronos_metrics, benchmark_metrics,
            save_path=save_path
        )
        
        assert Path(save_path).exists()
        plt.close(fig)
    
    def test_plot_rolling_sharpe(self, sample_backtest_results):
        """Verify rolling Sharpe plot works."""
        fig = plot_rolling_sharpe(sample_backtest_results, window=26)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ============================================================================
# ALLOCATION PLOTS TESTS
# ============================================================================

class TestAllocationPlots:
    """Tests for allocation visualization."""
    
    def test_plot_allocation_evolution(self, sample_backtest_results):
        """Verify allocation evolution stacked area chart works."""
        fig = plot_allocation_evolution(sample_backtest_results)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_regime_allocation_comparison(self, sample_backtest_results):
        """Verify regime allocation comparison works."""
        fig = plot_regime_allocation_comparison(sample_backtest_results)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_allocation_changes(self, sample_backtest_results):
        """Verify allocation changes plot works."""
        fig = plot_allocation_changes(sample_backtest_results)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestGenerateAllVisualizations:
    """Integration tests for full visualization generation."""
    
    def test_generate_all_visualizations_creates_files(self, sample_backtest_results, temp_output_dir):
        """Verify generate_all_visualizations creates expected files."""
        saved_files = generate_all_visualizations(
            sample_backtest_results,
            regime_detector=None,
            output_dir=temp_output_dir
        )
        
        # Should have created multiple files
        assert len(saved_files) > 0
        
        # Verify files exist
        for name, path in saved_files.items():
            assert Path(path).exists(), f"Expected {name} at {path}"
    
    def test_generate_all_visualizations_handles_missing_columns(self, temp_output_dir):
        """Verify graceful handling of missing columns."""
        # Create minimal DataFrame
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='B'),
            'portfolio_value': np.random.randn(10) * 100 + 100000
        })
        
        # Should not raise error
        saved_files = generate_all_visualizations(
            df,
            regime_detector=None,
            output_dir=temp_output_dir
        )
        
        # May have fewer files due to missing columns
        assert isinstance(saved_files, dict)


# ============================================================================
# COLOR CONSISTENCY TESTS
# ============================================================================

class TestColorConsistency:
    """Tests to verify color consistency across plots."""
    
    def test_regime_colors_match_characteristics(self):
        """Verify REGIME_COLORS match REGIME_CHARACTERISTICS."""
        from src.utils.regime_utils import REGIME_CHARACTERISTICS
        
        for regime_id in [0, 1, 2]:
            expected = REGIME_CHARACTERISTICS[regime_id]['hex_color']
            actual = REGIME_COLORS[regime_id]
            assert expected == actual, f"Color mismatch for regime {regime_id}"


# ============================================================================
# TEARDOWN
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_plots():
    """Close all matplotlib figures after each test."""
    yield
    plt.close('all')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
