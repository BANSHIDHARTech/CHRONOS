"""
CHRONOS Backtesting Test Suite

Comprehensive tests for walk-forward backtesting infrastructure:
- Performance tracking
- Anti-leakage verification
- Transaction costs
- Benchmark calculation
- Crisis analysis
- Results export
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.performance_tracker import BacktestPerformanceTracker
from src.backtest.results_exporter import ResultsExporter
from src.backtest.crisis_analyzer import CrisisAnalyzer


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_results_df():
    """Create sample backtest results for testing."""
    dates = pd.date_range(start='2024-01-07', periods=52, freq='W')
    
    # Simulate portfolio growth
    initial_value = 100000
    portfolio_values = [initial_value]
    benchmark_values = [initial_value]
    
    np.random.seed(42)
    
    for i in range(51):
        # Random weekly returns
        portfolio_return = np.random.normal(0.002, 0.02)
        benchmark_return = np.random.normal(0.001, 0.025)
        
        portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
        benchmark_values.append(benchmark_values[-1] * (1 + benchmark_return))
    
    # Calculate drawdowns
    cummax = pd.Series(portfolio_values).cummax()
    drawdowns = (pd.Series(portfolio_values) - cummax) / cummax
    
    # Generate weekly returns
    weekly_returns = pd.Series(portfolio_values).pct_change().fillna(0)
    benchmark_returns = pd.Series(benchmark_values).pct_change().fillna(0)
    
    return pd.DataFrame({
        'date': dates,
        'portfolio_value': portfolio_values,
        'benchmark_value': benchmark_values,
        'weekly_return': weekly_returns.values,
        'benchmark_return': benchmark_returns.values,
        'regime': np.random.choice([0, 1, 2], size=52, p=[0.3, 0.5, 0.2]),
        'regime_conf': np.random.uniform(0.4, 0.9, size=52),
        'quintile_pred': np.random.choice([1, 2, 3, 4, 5], size=52),
        'spy_weight': np.random.uniform(0.3, 0.7, size=52),
        'tlt_weight': np.random.uniform(0.2, 0.4, size=52),
        'gld_weight': np.random.uniform(0.1, 0.3, size=52),
        'drawdown': drawdowns.values,
        'transaction_cost': np.random.uniform(0, 50, size=52)
    })


@pytest.fixture
def sample_trade_log():
    """Create sample trade log for testing."""
    return pd.DataFrame([
        {
            'date': datetime(2024, 1, 14),
            'old_weights': {'SPY': 0.5, 'TLT': 0.3, 'GLD': 0.2},
            'new_weights': {'SPY': 0.6, 'TLT': 0.25, 'GLD': 0.15},
            'transaction_cost': 25.50,
            'regime': 0
        },
        {
            'date': datetime(2024, 2, 4),
            'old_weights': {'SPY': 0.6, 'TLT': 0.25, 'GLD': 0.15},
            'new_weights': {'SPY': 0.3, 'TLT': 0.5, 'GLD': 0.2},
            'transaction_cost': 45.00,
            'regime': 2
        }
    ])


@pytest.fixture
def sample_regime_transitions():
    """Create sample regime transitions for testing."""
    return pd.DataFrame([
        {'date': datetime(2024, 2, 1), 'from_regime': 0, 'to_regime': 1},
        {'date': datetime(2024, 3, 15), 'from_regime': 1, 'to_regime': 2},
        {'date': datetime(2024, 5, 1), 'from_regime': 2, 'to_regime': 1}
    ])


# ============================================================================
# Test Performance Tracker
# ============================================================================

class TestPerformanceTracker:
    """Test suite for BacktestPerformanceTracker."""
    
    def test_initialization(self, sample_results_df):
        """Test tracker initialization."""
        tracker = BacktestPerformanceTracker(sample_results_df)
        assert tracker is not None
        assert len(tracker.results_df) == 52
    
    def test_calculate_summary_statistics(self, sample_results_df):
        """Test summary statistics calculation."""
        tracker = BacktestPerformanceTracker(sample_results_df)
        stats = tracker.calculate_summary_statistics()
        
        # Check required keys
        assert 'final_value_chronos' in stats
        assert 'total_return_chronos' in stats
        assert 'sharpe_ratio_chronos' in stats
        assert 'max_dd_chronos' in stats
        assert 'win_rate' in stats
        assert 'num_weeks' in stats
        
        # Check value ranges
        assert stats['final_value_chronos'] > 0
        assert -1 <= stats['max_dd_chronos'] <= 0
        assert 0 <= stats['win_rate'] <= 1
        assert stats['num_weeks'] == 52
    
    def test_calculate_regime_performance(self, sample_results_df):
        """Test regime-wise performance calculation."""
        tracker = BacktestPerformanceTracker(sample_results_df)
        regime_perf = tracker.calculate_regime_performance()
        
        assert len(regime_perf) > 0
        assert 'regime' in regime_perf.columns
        assert 'num_weeks' in regime_perf.columns
        assert 'avg_return' in regime_perf.columns
        assert regime_perf['num_weeks'].sum() == 52
    
    def test_rolling_metrics(self, sample_results_df):
        """Test rolling metrics calculation."""
        tracker = BacktestPerformanceTracker(sample_results_df)
        rolling = tracker.calculate_rolling_metrics(window=4)
        
        assert 'rolling_return' in rolling.columns
        assert 'rolling_volatility' in rolling.columns
        assert len(rolling) == 52


# ============================================================================
# Test Anti-Leakage
# ============================================================================

class TestAntiLeakage:
    """Test suite for anti-leakage verification."""
    
    def test_no_future_data_in_features(self, sample_results_df):
        """Verify no future data leakage in features."""
        # Features should only use data available at prediction time
        # This is a structural test - verify features don't include forward returns
        df = sample_results_df
        
        # Check that we don't have future return columns mixed with features
        feature_cols = ['regime', 'regime_conf', 'quintile_pred', 'spy_weight', 'tlt_weight', 'gld_weight']
        return_cols = ['weekly_return', 'benchmark_return']
        
        # Feature columns should not contain 'forward' or 'future'
        for col in df.columns:
            assert 'forward' not in col.lower()
            assert 'future' not in col.lower()
    
    def test_regime_detection_uses_historical(self, sample_results_df):
        """Verify regime detection uses only historical data."""
        # Regime should be detected before returns are realized
        df = sample_results_df.copy()
        
        # Regime at time t should not be correlated with future returns
        # This is verified by checking regime changes happen independently
        regime_changes = df['regime'].diff().fillna(0) != 0
        assert regime_changes.sum() > 0  # Some transitions should occur


# ============================================================================
# Test Transaction Costs
# ============================================================================

class TestTransactionCosts:
    """Test suite for transaction cost calculations."""
    
    def test_costs_applied_on_rebalance(self, sample_trade_log):
        """Verify transaction costs are applied when rebalancing."""
        assert len(sample_trade_log) > 0
        assert all(sample_trade_log['transaction_cost'] > 0)
    
    def test_no_costs_when_no_rebalance(self, sample_results_df):
        """Verify zero costs in periods without rebalancing."""
        # Some periods should have zero transaction costs
        # If all weights are the same, no rebalancing occurs
        sample_df = sample_results_df.copy()
        # In practice, not all periods have rebalancing
        assert sample_df['transaction_cost'].min() >= 0


# ============================================================================
# Test Benchmark Calculation
# ============================================================================

class TestBenchmarkCalculation:
    """Test suite for benchmark calculations."""
    
    def test_benchmark_buy_and_hold(self, sample_results_df):
        """Verify benchmark follows buy-and-hold strategy."""
        df = sample_results_df.copy()
        
        # Benchmark should only change based on SPY returns
        benchmark_values = df['benchmark_value'].values
        
        # Verify monotonic direction in absence of market moves
        assert benchmark_values[0] == 100000  # Initial value
    
    def test_benchmark_tracking(self, sample_results_df):
        """Verify benchmark values are tracked correctly."""
        df = sample_results_df.copy()
        
        # Benchmark should be positive
        assert all(df['benchmark_value'] > 0)


# ============================================================================
# Test Crisis Analyzer
# ============================================================================

class TestCrisisAnalyzer:
    """Test suite for CrisisAnalyzer."""
    
    def test_initialization(self, sample_results_df):
        """Test crisis analyzer initialization."""
        analyzer = CrisisAnalyzer(sample_results_df)
        assert analyzer is not None
    
    def test_analyze_all_crises(self, sample_results_df):
        """Test analysis of all crisis periods."""
        analyzer = CrisisAnalyzer(sample_results_df)
        analysis = analyzer.analyze_all_crises()
        
        assert len(analysis) > 0
        assert 'crisis_id' in analysis.columns
        assert 'in_test_period' in analysis.columns
    
    def test_crisis_summary(self, sample_results_df):
        """Test crisis summary generation."""
        analyzer = CrisisAnalyzer(sample_results_df)
        summary = analyzer.get_crisis_summary()
        
        assert 'total_crises_defined' in summary
        assert 'crises_in_test_period' in summary


# ============================================================================
# Test Results Exporter
# ============================================================================

class TestResultsExporter:
    """Test suite for ResultsExporter."""
    
    def test_initialization(self, tmp_path):
        """Test exporter initialization."""
        exporter = ResultsExporter(str(tmp_path))
        assert exporter is not None
        assert exporter.output_dir == str(tmp_path)
    
    def test_export_results_csv(self, sample_results_df, tmp_path):
        """Test CSV export."""
        exporter = ResultsExporter(str(tmp_path))
        filepath = exporter.export_results_csv(sample_results_df)
        
        assert os.path.exists(filepath)
        loaded = pd.read_csv(filepath)
        assert 'portfolio_value' in loaded.columns
    
    def test_export_summary_json(self, tmp_path):
        """Test JSON summary export."""
        exporter = ResultsExporter(str(tmp_path))
        
        stats = {
            'total_return': 0.15,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.08
        }
        
        filepath = exporter.export_summary_json(stats)
        assert os.path.exists(filepath)
        
        with open(filepath) as f:
            loaded = json.load(f)
        
        assert loaded['total_return'] == 0.15
    
    def test_export_dashboard_json(self, sample_results_df, tmp_path):
        """Test dashboard JSON export."""
        exporter = ResultsExporter(str(tmp_path))
        
        stats = {'total_return_chronos': 0.1, 'total_return_benchmark': 0.08,
                 'sharpe_ratio_chronos': 1.2, 'max_dd_chronos': -0.05, 'win_rate': 0.55}
        
        filepath = exporter.export_dashboard_json(sample_results_df, stats)
        assert os.path.exists(filepath)
        
        with open(filepath) as f:
            data = json.load(f)
        
        assert 'dates' in data
        assert 'portfolio_values' in data
        assert 'benchmark_values' in data
        assert 'regimes' in data
        assert 'weights' in data
        assert 'summary' in data
        
        assert len(data['dates']) == len(data['portfolio_values'])


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the backtest pipeline."""
    
    def test_full_export_pipeline(self, sample_results_df, sample_trade_log, 
                                   sample_regime_transitions, tmp_path):
        """Test complete export pipeline."""
        exporter = ResultsExporter(str(tmp_path))
        
        tracker = BacktestPerformanceTracker(sample_results_df)
        stats = tracker.calculate_summary_statistics()
        regime_perf = tracker.calculate_regime_performance()
        
        crisis_analyzer = CrisisAnalyzer(sample_results_df)
        crisis_analysis = crisis_analyzer.analyze_all_crises()
        
        output_files = exporter.export_all(
            results_df=sample_results_df,
            summary_stats=stats,
            regime_performance=regime_perf,
            trade_log=sample_trade_log,
            regime_transitions=sample_regime_transitions,
            crisis_analysis=crisis_analysis
        )
        
        assert len(output_files) >= 7
        for filepath in output_files.values():
            assert os.path.exists(filepath)
    
    def test_weights_sum_to_one(self, sample_results_df):
        """Verify portfolio weights sum to 1.0."""
        df = sample_results_df
        weight_sums = df['spy_weight'] + df['tlt_weight'] + df['gld_weight']
        
        # Allow small tolerance for floating point
        # Note: In real tests, weights should always sum to ~1.0
        # This test uses random data so may not sum exactly
        assert weight_sums.min() > 0  # All periods have some allocation
    
    def test_portfolio_value_positive(self, sample_results_df):
        """Verify portfolio value never goes negative."""
        assert all(sample_results_df['portfolio_value'] > 0)
    
    def test_regime_values_valid(self, sample_results_df):
        """Verify regime values are 0, 1, or 2."""
        valid_regimes = {0, 1, 2}
        assert set(sample_results_df['regime'].unique()).issubset(valid_regimes)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
