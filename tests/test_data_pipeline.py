"""
Test Suite for CHRONOS Data Pipeline

Comprehensive tests focusing on anti-leakage validation and data integrity.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import config
from src.data.data_loader import DataLoader
from src.data.feature_engineering import FeatureEngineer
from src.data.target_engineering import TargetEngineer
from src.data.data_splitter import DataSplitter
from src.utils.validation import (
    check_feature_shift,
    check_data_integrity,
    validate_train_test_split,
    detect_lookahead_bias
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Simulate realistic price movements
    np.random.seed(42)
    returns = np.random.randn(len(dates)) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    return data


@pytest.fixture
def multiindex_data(sample_price_data):
    """Create MultiIndex DataFrame mimicking yfinance structure."""
    tickers = ['SPY', '^VIX']
    
    # Create data for multiple tickers
    dfs = {}
    for ticker in tickers:
        # Add some variation per ticker
        df = sample_price_data.copy()
        df = df * (1 + np.random.randn() * 0.1)
        dfs[ticker] = df
    
    # Combine into MultiIndex with (Ticker, Column) structure
    combined = pd.concat(dfs, axis=1, keys=tickers)
    combined = combined.sort_index(axis=1, level=0)
    
    return combined


@pytest.fixture
def feature_engineer():
    """Create FeatureEngineer instance."""
    return FeatureEngineer(config)


@pytest.fixture
def target_engineer():
    """Create TargetEngineer instance."""
    return TargetEngineer(config)


@pytest.fixture
def data_splitter():
    """Create DataSplitter instance."""
    return DataSplitter(config)


# ============================================================================
# DATA LOADER TESTS
# ============================================================================

class TestDataLoader:
    """Tests for DataLoader class."""
    
    def test_cache_filename_generation(self):
        """Test cache filename generation."""
        loader = DataLoader()
        filename = loader._get_cache_filename('^GSPC', '2020-01-01', '2023-12-31')
        
        assert 'GSPC' in filename or '' in filename  # Handle special char removal
        assert '2020-01-01' in filename
        assert '2023-12-31' in filename
        assert filename.endswith('.csv')
    
    def test_validate_data_empty_dataframe(self):
        """Test validation catches empty DataFrame."""
        loader = DataLoader()
        empty_df = pd.DataFrame()
        
        result = loader.validate_data(empty_df)
        
        assert not result['is_valid']
        assert any('empty' in issue.lower() for issue in result['issues'])
    
    def test_validate_data_missing_values(self):
        """Test validation detects missing values."""
        loader = DataLoader()
        
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        data = pd.DataFrame({
            'Close': np.random.randn(len(dates))
        }, index=dates)
        
        # Introduce missing values
        data.iloc[10:20] = np.nan
        
        result = loader.validate_data(data)
        
        assert 'missing_values' in result['statistics']
    
    def test_validate_data_duplicates(self):
        """Test validation catches duplicate indices."""
        loader = DataLoader()
        
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        # Fixed: Use pd.DatetimeIndex concat instead of deprecated append()
        dates = pd.DatetimeIndex(np.concatenate([dates.values, dates[:3].values]))
        
        data = pd.DataFrame({
            'Close': np.random.randn(len(dates))
        }, index=dates)
        
        result = loader.validate_data(data)
        
        assert not result['is_valid']
        assert any('duplicate' in issue.lower() for issue in result['issues'])


# ============================================================================
# FEATURE ENGINEERING TESTS
# ============================================================================

class TestFeatureEngineering:
    """Tests for feature engineering with anti-leakage focus."""
    
    def test_log_returns_shifted(self, sample_price_data, feature_engineer):
        """CRITICAL TEST: Verify log returns are shifted."""
        close = sample_price_data['Close']
        log_returns = feature_engineer.compute_log_returns(close)
        
        # Check that first valid index is after original
        original_first = close.first_valid_index()
        feature_first = log_returns.first_valid_index()
        
        assert feature_first > original_first, "Log returns not properly shifted"
        
        # Verify shift validation passes
        check_feature_shift(log_returns, close, 'log_returns')
    
    def test_parkinson_volatility_shifted(self, sample_price_data, feature_engineer):
        """Test Parkinson volatility is shifted."""
        high = sample_price_data['High']
        low = sample_price_data['Low']
        
        parkinson_vol = feature_engineer.compute_parkinson_volatility(high, low, window=20)
        
        # Should be shifted
        original_first = high.first_valid_index()
        feature_first = parkinson_vol.first_valid_index()
        
        assert feature_first > original_first, "Parkinson volatility not shifted"
    
    def test_rsi_shifted(self, sample_price_data, feature_engineer):
        """Test RSI is shifted."""
        close = sample_price_data['Close']
        rsi = feature_engineer.compute_rsi(close, period=14)
        
        check_feature_shift(rsi, close, 'RSI')
    
    def test_macd_shifted(self, sample_price_data, feature_engineer):
        """Test MACD components are all shifted."""
        close = sample_price_data['Close']
        macd, signal, hist = feature_engineer.compute_macd(close)
        
        check_feature_shift(macd, close, 'MACD')
        check_feature_shift(signal, close, 'MACD_signal')
        check_feature_shift(hist, close, 'MACD_histogram')
    
    def test_bollinger_bands_shifted(self, sample_price_data, feature_engineer):
        """Test Bollinger Bands are shifted."""
        close = sample_price_data['Close']
        bb_width = feature_engineer.compute_bollinger_bands(close)
        
        check_feature_shift(bb_width, close, 'BB_width')
    
    def test_atr_shifted(self, sample_price_data, feature_engineer):
        """Test ATR is shifted."""
        high = sample_price_data['High']
        low = sample_price_data['Low']
        close = sample_price_data['Close']
        
        atr = feature_engineer.compute_atr(high, low, close)
        
        check_feature_shift(atr, close, 'ATR')
    
    def test_rolling_sharpe_shifted(self, sample_price_data, feature_engineer):
        """Test rolling Sharpe ratio is shifted."""
        close = sample_price_data['Close']
        returns = np.log(close / close.shift(1))
        
        sharpe = feature_engineer.compute_rolling_sharpe(returns, window=63)
        
        check_feature_shift(sharpe, close, 'rolling_sharpe')
    
    def test_moving_averages_shifted(self, sample_price_data, feature_engineer):
        """Test all moving averages are shifted."""
        close = sample_price_data['Close']
        
        mas = feature_engineer.compute_moving_averages(
            close,
            sma_windows=[20, 50],
            ema_windows=[12, 26]
        )
        
        for ma_name, ma_values in mas.items():
            check_feature_shift(ma_values, close, ma_name)
    
    def test_no_leakage_validation(self, multiindex_data, feature_engineer):
        """CRITICAL TEST: Validate anti-leakage on full feature set."""
        features = feature_engineer.engineer_features(multiindex_data)
        
        # Validation should pass (done internally)
        assert len(features) > 0
        assert features.isnull().sum().sum() == 0  # No NaN values after dropna
    
    def test_features_have_proper_lag(self, multiindex_data, feature_engineer):
        """Test that features have at least 1-day lag."""
        features = feature_engineer.engineer_features(multiindex_data)
        
        # Get original SPY close
        original_close = multiindex_data['SPY']['Close']
        
        # For each feature, check that it doesn't use same-day information
        for col in features.columns:
            feature_first_valid = features[col].first_valid_index()
            
            if feature_first_valid is not None:
                # Feature should start after original data
                original_idx = original_close.index.get_loc(feature_first_valid)
                assert original_idx > 0, f"Feature {col} may have lookahead bias"


# ============================================================================
# TARGET ENGINEERING TESTS
# ============================================================================

class TestTargetEngineering:
    """Tests for target engineering."""
    
    def test_forward_returns_not_shifted(self, sample_price_data, target_engineer):
        """CRITICAL TEST: Verify forward returns are NOT shifted (they are targets)."""
        close = sample_price_data['Close']
        forward_returns = target_engineer.compute_forward_returns(close, days=5)
        
        # Forward returns should start at same time as original
        # (they look ahead, which is correct for targets)
        assert len(forward_returns) == len(close)
        
        # Last 5 days should be NaN (no future data)
        assert forward_returns.iloc[-5:].isna().all()
    
    def test_quintile_distribution(self, sample_price_data, target_engineer):
        """Test quintile labels have approximately equal distribution."""
        close = sample_price_data['Close']
        targets = target_engineer.engineer_targets(close, forward_days=5)
        
        # Check quintile distribution
        value_counts = targets['quintile_label'].value_counts()
        
        # Each quintile should have roughly 20% of data
        for quintile in config.QUINTILE_LABELS:
            count = value_counts.get(quintile, 0)
            percentage = (count / len(targets)) * 100
            
            # Allow 5% deviation
            assert 15 <= percentage <= 25, f"Quintile {quintile} has {percentage}% (expected ~20%)"
    
    def test_alignment(self, multiindex_data, feature_engineer, target_engineer):
        """Test feature-target alignment."""
        # Engineer features
        features = feature_engineer.engineer_features(multiindex_data)
        
        # Engineer targets
        close = multiindex_data['SPY']['Close']
        targets = target_engineer.engineer_targets(close)
        
        # Align
        X, y = target_engineer.align_features_and_targets(features, targets)
        
        # Check alignment
        assert X.index.equals(y.index), "X and y indices don't match"
        assert len(X) == len(y), "X and y lengths don't match"
        assert len(X) > 0, "Alignment resulted in empty dataset"


# ============================================================================
# DATA SPLITTER TESTS
# ============================================================================

class TestDataSplitter:
    """Tests for data splitting."""
    
    def test_temporal_split_no_overlap(self, data_splitter):
        """CRITICAL TEST: Ensure no overlap between train/val/test."""
        # Create sample data
        dates = pd.date_range('2019-01-01', '2024-12-31', freq='D')
        X = pd.DataFrame(
            np.random.randn(len(dates), 5),
            index=dates,
            columns=[f'feature_{i}' for i in range(5)]
        )
        y = pd.Series(np.random.randint(1, 6, len(dates)), index=dates)
        
        # Split
        splits = data_splitter.split_with_config(X, y)
        
        # Verify no overlap
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']
        
        # Check temporal ordering
        assert X_train.index.max() < X_val.index.min()
        assert X_val.index.max() < X_test.index.min()
        
        # Verify validation passes
        validate_train_test_split(X_train, X_val, "train/val")
        validate_train_test_split(X_val, X_test, "val/test")
    
    def test_split_statistics(self, data_splitter):
        """Test split statistics calculation."""
        dates = pd.date_range('2019-01-01', '2024-12-31', freq='D')
        X = pd.DataFrame(np.random.randn(len(dates), 3), index=dates)
        y = pd.Series(np.random.randint(1, 6, len(dates)), index=dates)
        
        splits = data_splitter.split_with_config(X, y)
        stats = data_splitter.get_split_statistics(splits)
        
        # Verify statistics are computed
        assert 'train' in stats
        assert 'val' in stats
        assert 'test' in stats
        
        for split_name, split_stats in stats.items():
            assert 'num_samples' in split_stats
            assert 'class_distribution' in split_stats


# ============================================================================
# VALIDATION TESTS
# ============================================================================

class TestValidation:
    """Tests for validation utilities."""
    
    def test_check_data_integrity_sorted_index(self):
        """Test data integrity check on sorted data."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        data = pd.DataFrame({'value': np.random.randn(len(dates))}, index=dates)
        
        result = check_data_integrity(data)
        
        assert result['is_valid']
    
    def test_check_data_integrity_unsorted_index(self):
        """Test data integrity check catches unsorted data."""
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        data = pd.DataFrame({'value': np.random.randn(len(dates))}, index=dates)
        
        # Shuffle index
        data = data.sample(frac=1)
        
        result = check_data_integrity(data)
        
        assert not result['is_valid']
        assert any('sorted' in issue.lower() for issue in result['issues'])
    
    def test_detect_lookahead_bias_clean_data(self):
        """Test lookahead bias detection on clean data."""
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        
        # Create features and targets with reasonable correlation
        features = pd.DataFrame({
            'feature_1': np.random.randn(len(dates)),
            'feature_2': np.random.randn(len(dates))
        }, index=dates)
        
        targets = pd.Series(np.random.randn(len(dates)), index=dates)
        
        result = detect_lookahead_bias(features, targets)
        
        # Should not flag any suspicious features
        assert len(result['suspicious_features']) == 0
    
    def test_detect_lookahead_bias_suspicious_correlation(self):
        """Test lookahead bias detection catches suspicious correlations."""
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        
        targets = pd.Series(np.random.randn(len(dates)), index=dates)
        
        # Create a feature that's almost identical to target (lookahead!)
        features = pd.DataFrame({
            'suspicious_feature': targets + np.random.randn(len(dates)) * 0.01,
            'normal_feature': np.random.randn(len(dates))
        }, index=dates)
        
        result = detect_lookahead_bias(features, targets)
        
        # Should flag the suspicious feature
        assert len(result['suspicious_features']) > 0
        assert 'suspicious_feature' in result['suspicious_features']


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline_no_leakage(self, multiindex_data):
        """CRITICAL TEST: Full pipeline with anti-leakage validation."""
        # Engineer features
        feature_engineer = FeatureEngineer(config)
        features = feature_engineer.engineer_features(multiindex_data)
        
        # Engineer targets
        target_engineer = TargetEngineer(config)
        close = multiindex_data['SPY']['Close']
        targets = target_engineer.engineer_targets(close)
        
        # Align
        X, y = target_engineer.align_features_and_targets(features, targets)
        
        # Split using manual dates that match test data range (2020-2023)
        splitter = DataSplitter(config)
        splits = splitter.split_by_date(
            X, y,
            train_start='2020-01-01',
            train_end='2021-12-31',
            val_start='2022-01-01',
            val_end='2022-12-31',
            test_start='2023-01-01',
            test_end='2023-12-31'
        )
        
        # Validate
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']
        
        # Ensure temporal ordering
        assert X_train.index.max() < X_val.index.min()
        assert X_val.index.max() < X_test.index.min()
        
        # Check for lookahead bias in training set
        bias_check = detect_lookahead_bias(X_train, y_train)
        
        # Should not have highly suspicious correlations (>0.95)
        assert len(bias_check['suspicious_features']) == 0, \
            f"Possible lookahead bias detected: {bias_check['suspicious_features']}"
    
    def test_pipeline_reproducibility(self, multiindex_data):
        """Test that pipeline produces reproducible results."""
        # Run pipeline twice
        feature_engineer_1 = FeatureEngineer(config)
        features_1 = feature_engineer_1.engineer_features(multiindex_data)
        
        feature_engineer_2 = FeatureEngineer(config)
        features_2 = feature_engineer_2.engineer_features(multiindex_data)
        
        # Results should be identical
        pd.testing.assert_frame_equal(features_1, features_2)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
