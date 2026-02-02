"""
Unit Tests for CHRONOS Streamlit Dashboard

Tests data loading functions, visualization generators,
and scenario analyzer with mock data.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_results_df():
    """Create sample backtest results DataFrame."""
    dates = pd.date_range('2024-01-01', periods=252, freq='B')
    n = len(dates)
    
    np.random.seed(42)
    
    # Simulate portfolio value with some drift
    returns = 0.0003 + np.random.randn(n) * 0.01
    portfolio_value = 100000 * np.cumprod(1 + returns)
    
    # Simulate benchmark
    benchmark_returns = 0.0002 + np.random.randn(n) * 0.012
    benchmark_value = 100000 * np.cumprod(1 + benchmark_returns)
    
    # Simulate regimes (with some persistence)
    regimes = []
    current = 1
    for _ in range(n):
        if np.random.random() < 0.05:
            current = np.random.choice([0, 1, 2])
        regimes.append(current)
    
    df = pd.DataFrame({
        'portfolio_value': portfolio_value,
        'benchmark_value': benchmark_value,
        'regime': regimes,
        'weight_SPY': np.random.uniform(0.3, 0.7, n),
        'weight_TLT': np.random.uniform(0.2, 0.4, n),
        'weight_GLD': np.random.uniform(0.1, 0.3, n),
        'weekly_return': returns
    }, index=dates)
    
    # Normalize weights
    total_weights = df['weight_SPY'] + df['weight_TLT'] + df['weight_GLD']
    df['weight_SPY'] /= total_weights
    df['weight_TLT'] /= total_weights
    df['weight_GLD'] /= total_weights
    
    return df


@pytest.fixture
def sample_summary_stats():
    """Create sample summary statistics."""
    return {
        'total_return': 0.15,
        'annualized_return': 0.12,
        'sharpe_ratio': 1.5,
        'sortino_ratio': 2.1,
        'max_drawdown': -0.08,
        'win_rate': 0.55,
        'volatility': 0.12,
        'calmar_ratio': 1.5,
        'final_portfolio_value': 115000
    }


@pytest.fixture
def sample_transition_matrix():
    """Create sample transition probability matrix."""
    return np.array([
        [0.85, 0.10, 0.05],
        [0.15, 0.70, 0.15],
        [0.10, 0.20, 0.70]
    ])


# ============================================================================
# STREAMLIT HELPERS TESTS
# ============================================================================

class TestStreamlitHelpers:
    """Tests for streamlit_helpers.py functions."""
    
    def test_format_percentage(self):
        """Test percentage formatting."""
        from src.utils.streamlit_helpers import format_percentage
        
        assert format_percentage(0.15) == "15.00%"
        assert format_percentage(0.1567, decimals=1) == "15.7%"
        assert format_percentage(-0.05) == "-5.00%"
    
    def test_format_currency(self):
        """Test currency formatting."""
        from src.utils.streamlit_helpers import format_currency
        
        assert format_currency(100000) == "$100,000.00"
        assert format_currency(1234.5, decimals=0) == "$1,235"
    
    def test_get_regime_name(self):
        """Test regime name lookup."""
        from src.utils.streamlit_helpers import get_regime_name
        
        assert get_regime_name(0) == "Euphoria"
        assert get_regime_name(1) == "Complacency"
        assert get_regime_name(2) == "Capitulation"
        assert get_regime_name(99) == "Unknown"
    
    def test_get_regime_color(self):
        """Test regime color lookup."""
        from src.utils.streamlit_helpers import get_regime_color
        
        assert get_regime_color(0) == "#00C853"  # Green
        assert get_regime_color(1) == "#FFD600"  # Yellow
        assert get_regime_color(2) == "#D50000"  # Red
    
    def test_export_results_for_download(self, sample_results_df):
        """Test CSV export function."""
        from src.utils.streamlit_helpers import export_results_for_download
        
        csv_bytes = export_results_for_download(sample_results_df)
        
        assert isinstance(csv_bytes, bytes)
        assert len(csv_bytes) > 0
        assert b'portfolio_value' in csv_bytes
    
    def test_export_summary_for_download(self, sample_summary_stats):
        """Test JSON export function."""
        from src.utils.streamlit_helpers import export_summary_for_download
        
        json_bytes = export_summary_for_download(sample_summary_stats)
        
        assert isinstance(json_bytes, bytes)
        parsed = json.loads(json_bytes.decode('utf-8'))
        assert parsed['sharpe_ratio'] == 1.5


# ============================================================================
# BACKTEST RUNNER TESTS
# ============================================================================

class TestBacktestRunner:
    """Tests for backtest_runner.py functions."""
    
    def test_validate_date_range_valid(self):
        """Test valid date range validation."""
        from src.utils.backtest_runner import validate_date_range
        
        is_valid, msg = validate_date_range('2024-01-01', '2024-12-31')
        
        assert is_valid is True
        assert "Valid" in msg
    
    def test_validate_date_range_invalid_order(self):
        """Test invalid date order."""
        from src.utils.backtest_runner import validate_date_range
        
        is_valid, msg = validate_date_range('2024-12-31', '2024-01-01')
        
        assert is_valid is False
        assert "before" in msg.lower()
    
    def test_validate_date_range_too_short(self):
        """Test date range too short."""
        from src.utils.backtest_runner import validate_date_range
        
        is_valid, msg = validate_date_range('2024-01-01', '2024-01-15')
        
        assert is_valid is False
        assert "30 days" in msg
    
    def test_get_rebalance_frequency_options(self):
        """Test rebalance frequency options."""
        from src.utils.backtest_runner import get_rebalance_frequency_options
        
        options = get_rebalance_frequency_options()
        
        assert 'weekly' in options
        assert 'monthly' in options
        assert len(options) >= 3
    
    def test_format_backtest_summary(self, sample_summary_stats):
        """Test summary formatting."""
        from src.utils.backtest_runner import format_backtest_summary
        
        df = format_backtest_summary(sample_summary_stats)
        
        assert isinstance(df, pd.DataFrame)
        assert 'Metric' in df.columns
        assert 'Value' in df.columns
        assert len(df) > 0


# ============================================================================
# SCENARIO ANALYZER TESTS
# ============================================================================

class TestScenarioAnalyzer:
    """Tests for scenario_analyzer.py functions."""
    
    def test_get_regime_allocation(self):
        """Test regime allocation lookup."""
        from src.utils.scenario_analyzer import get_regime_allocation
        
        alloc = get_regime_allocation(0)  # Euphoria
        
        assert 'SPY' in alloc
        assert 'TLT' in alloc
        assert 'GLD' in alloc
        assert abs(sum(alloc.values()) - 1.0) < 0.01  # Sums to 1
        
        # Euphoria should have higher SPY allocation
        assert alloc['SPY'] > alloc['TLT']
    
    def test_get_regime_allocation_capitulation(self):
        """Test capitulation regime allocation."""
        from src.utils.scenario_analyzer import get_regime_allocation
        
        alloc = get_regime_allocation(2)  # Capitulation
        
        # Capitulation should have lower SPY, higher TLT
        assert alloc['TLT'] > alloc['SPY']
    
    def test_analyze_regime_override(self):
        """Test regime override analysis."""
        from src.utils.scenario_analyzer import analyze_regime_override
        
        result = analyze_regime_override(
            current_regime=1,  # Complacency
            override_regime=0,  # Euphoria
            current_weights={'SPY': 0.5, 'TLT': 0.3, 'GLD': 0.2},
            confidence=0.8
        )
        
        assert 'current_allocation' in result
        assert 'override_allocation' in result
        assert 'weight_changes' in result
        assert 'risk_impact' in result
        assert result['current_regime'] == 1
        assert result['override_regime'] == 0
    
    def test_get_scenario_comparison_table(self):
        """Test scenario comparison table generation."""
        from src.utils.scenario_analyzer import (
            analyze_regime_override, get_scenario_comparison_table
        )
        
        analysis = analyze_regime_override(
            current_regime=1,
            override_regime=2,
            current_weights={'SPY': 0.5, 'TLT': 0.3, 'GLD': 0.2}
        )
        
        table = get_scenario_comparison_table(analysis)
        
        assert isinstance(table, pd.DataFrame)
        assert 'Asset' in table.columns
        assert 'Change' in table.columns
        assert len(table) == 3  # SPY, TLT, GLD
    
    def test_get_regime_recommendation_high_confidence(self):
        """Test regime recommendation with high confidence."""
        from src.utils.scenario_analyzer import get_regime_recommendation
        
        rec = get_regime_recommendation(
            current_regime=1,
            model_confidence=0.85
        )
        
        assert 'current_regime' in rec
        assert 'model_confidence' in rec
        assert rec['override_suggested'] is False
        assert "High confidence" in rec['reason']
    
    def test_get_regime_recommendation_low_confidence(self):
        """Test regime recommendation with low confidence."""
        from src.utils.scenario_analyzer import get_regime_recommendation
        
        rec = get_regime_recommendation(
            current_regime=1,
            model_confidence=0.35
        )
        
        assert rec['override_suggested'] is True
        assert "Low" in rec['reason']


# ============================================================================
# PLOTLY CHARTS TESTS
# ============================================================================

class TestPlotlyCharts:
    """Tests for plotly_charts.py visualization functions."""
    
    def test_create_interactive_regime_chart(self, sample_results_df):
        """Test interactive regime chart creation."""
        from src.visualization.plotly_charts import create_interactive_regime_chart
        import plotly.graph_objects as go
        
        fig = create_interactive_regime_chart(
            prices=sample_results_df['portfolio_value'],
            regimes=sample_results_df['regime'],
            dates=sample_results_df.index,
            title="Test Chart"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_create_interactive_performance_chart(self, sample_results_df):
        """Test performance chart creation."""
        from src.visualization.plotly_charts import create_interactive_performance_chart
        import plotly.graph_objects as go
        
        fig = create_interactive_performance_chart(
            chronos_values=sample_results_df['portfolio_value'],
            benchmark_values=sample_results_df['benchmark_value'],
            dates=sample_results_df.index
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # At least cumulative returns for both
    
    def test_create_interactive_allocation_chart(self, sample_results_df):
        """Test allocation chart creation."""
        from src.visualization.plotly_charts import create_interactive_allocation_chart
        import plotly.graph_objects as go
        
        fig = create_interactive_allocation_chart(sample_results_df)
        
        assert isinstance(fig, go.Figure)
    
    def test_create_interactive_transition_heatmap(self, sample_transition_matrix):
        """Test transition heatmap creation."""
        from src.visualization.plotly_charts import create_interactive_transition_heatmap
        import plotly.graph_objects as go
        
        fig = create_interactive_transition_heatmap(sample_transition_matrix)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # One heatmap trace
    
    def test_create_rolling_metrics_chart(self, sample_results_df):
        """Test rolling metrics chart creation."""
        from src.visualization.plotly_charts import create_rolling_metrics_chart
        import plotly.graph_objects as go
        
        fig = create_rolling_metrics_chart(sample_results_df, window=13)
        
        assert isinstance(fig, go.Figure)
    
    def test_create_monthly_returns_heatmap(self, sample_results_df):
        """Test monthly returns heatmap creation."""
        from src.visualization.plotly_charts import create_monthly_returns_heatmap
        import plotly.graph_objects as go
        
        fig = create_monthly_returns_heatmap(sample_results_df)
        
        assert isinstance(fig, go.Figure)


# ============================================================================
# CUSTOM CSS TESTS
# ============================================================================

class TestCustomCSS:
    """Tests for custom_css.py styling functions."""
    
    def test_get_custom_css(self):
        """Test CSS generation."""
        from src.visualization.custom_css import get_custom_css
        
        css = get_custom_css()
        
        assert isinstance(css, str)
        assert '<style>' in css
        assert '</style>' in css
        assert '.regime-badge' in css
        assert '.metric-card' in css
    
    def test_get_regime_badge_html(self):
        """Test regime badge HTML generation."""
        from src.visualization.custom_css import get_regime_badge_html
        
        html = get_regime_badge_html(0, confidence=0.85)
        
        assert 'EUPHORIA' in html
        assert '85%' in html
        assert 'regime-euphoria' in html
    
    def test_get_allocation_bar_html(self):
        """Test allocation bar HTML generation."""
        from src.visualization.custom_css import get_allocation_bar_html
        
        html = get_allocation_bar_html(0.5, 0.3, 0.2)
        
        assert 'SPY' in html
        assert 'TLT' in html
        assert 'GLD' in html
        assert '50%' in html


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        from src.visualization.plotly_charts import create_interactive_allocation_chart
        import plotly.graph_objects as go
        
        empty_df = pd.DataFrame()
        fig = create_interactive_allocation_chart(empty_df)
        
        assert isinstance(fig, go.Figure)
    
    def test_single_data_point(self):
        """Test handling of single data point."""
        from src.utils.scenario_analyzer import get_regime_allocation
        
        # Should work with any regime
        for regime in [0, 1, 2]:
            alloc = get_regime_allocation(regime)
            assert sum(alloc.values()) > 0
    
    def test_invalid_regime_id(self):
        """Test handling of invalid regime ID."""
        from src.utils.streamlit_helpers import get_regime_name
        
        # Should return 'Unknown' for invalid regime
        assert get_regime_name(99) == 'Unknown'
        assert get_regime_name(-1) == 'Unknown'


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
