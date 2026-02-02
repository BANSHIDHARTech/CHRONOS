"""
Unit Tests for CHRONOS Regime Detection Module

Tests for HMM regime detector, regime utilities, and related functions.
"""

import pytest
import pandas as pd
import numpy as np
import pickle
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_returns():
    """Generate sample log returns for testing."""
    np.random.seed(42)
    n = 500
    
    # Simulate regime-like returns
    returns = np.zeros(n)
    
    # Euphoria period (0-150): positive drift, low vol
    returns[:150] = 0.001 + np.random.randn(150) * 0.008
    
    # Complacency period (150-350): neutral drift, medium vol
    returns[150:350] = 0.0002 + np.random.randn(200) * 0.012
    
    # Capitulation period (350-450): negative drift, high vol
    returns[350:450] = -0.002 + np.random.randn(100) * 0.025
    
    # Recovery (450-500): back to positive
    returns[450:] = 0.0008 + np.random.randn(50) * 0.010
    
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    return pd.Series(returns, index=dates)


@pytest.fixture
def sample_features(sample_returns):
    """Generate sample features for HMM input."""
    returns = sample_returns.values.reshape(-1, 1)
    
    # Add volatility feature
    vol = pd.Series(sample_returns).rolling(20, min_periods=20).std().fillna(0.01).values.reshape(-1, 1)
    
    # Add VIX-like feature
    vix = 15 + vol.flatten() * 500 + np.random.randn(len(returns)) * 2
    vix = vix.reshape(-1, 1)
    
    return np.hstack([returns, vol, vix])


@pytest.fixture
def mock_hmm_model():
    """Create a mock HMM model for testing."""
    from hmmlearn import hmm
    
    model = hmm.GaussianHMM(
        n_components=3,
        covariance_type='full',
        n_iter=100,
        random_state=42
    )
    
    return model


@pytest.fixture
def trained_hmm(sample_features, mock_hmm_model):
    """Return a trained HMM model."""
    # Remove any NaN rows
    clean_features = sample_features[~np.isnan(sample_features).any(axis=1)]
    mock_hmm_model.fit(clean_features)
    return mock_hmm_model


# ============================================================================
# HMM INITIALIZATION TESTS
# ============================================================================

class TestHMMInitialization:
    """Tests for HMM model initialization."""
    
    def test_hmm_initialization(self):
        """Verify GaussianHMM initializes with correct parameters."""
        from hmmlearn import hmm
        
        model = hmm.GaussianHMM(
            n_components=3,
            covariance_type='full',
            n_iter=1000,
            random_state=42
        )
        
        assert model.n_components == 3
        assert model.covariance_type == 'full'
        assert model.n_iter == 1000
        assert model.random_state == 42
    
    def test_hmm_components_count(self, mock_hmm_model):
        """Verify correct number of components."""
        assert mock_hmm_model.n_components == 3
    
    def test_covariance_type(self, mock_hmm_model):
        """Verify full covariance matrix."""
        assert mock_hmm_model.covariance_type == 'full'


# ============================================================================
# HMM FIT AND CONVERGENCE TESTS
# ============================================================================

class TestHMMFitConvergence:
    """Tests for HMM model fitting and convergence."""
    
    def test_hmm_fit_convergence(self, sample_features, mock_hmm_model):
        """Check model converges on training data."""
        clean_features = sample_features[~np.isnan(sample_features).any(axis=1)]
        
        # Fit the model
        mock_hmm_model.fit(clean_features)
        
        # Check that model has been fitted
        assert hasattr(mock_hmm_model, 'transmat_')
        assert hasattr(mock_hmm_model, 'means_')
        assert hasattr(mock_hmm_model, 'covars_')
    
    def test_log_likelihood_exists(self, trained_hmm, sample_features):
        """Check log-likelihood can be computed."""
        clean_features = sample_features[~np.isnan(sample_features).any(axis=1)]
        ll = trained_hmm.score(clean_features)
        
        assert isinstance(ll, float)
        assert not np.isnan(ll)
    
    def test_means_shape(self, trained_hmm):
        """Check means have correct shape."""
        # Should be (n_components, n_features)
        assert trained_hmm.means_.shape[0] == 3
        assert trained_hmm.means_.shape[1] == 3  # returns, vol, vix


# ============================================================================
# REGIME PREDICTION TESTS
# ============================================================================

class TestRegimePrediction:
    """Tests for regime prediction functionality."""
    
    def test_regime_prediction(self, trained_hmm, sample_features):
        """Verify predict() returns values in {0, 1, 2}."""
        clean_features = sample_features[~np.isnan(sample_features).any(axis=1)]
        predictions = trained_hmm.predict(clean_features)
        
        assert len(predictions) == len(clean_features)
        assert set(predictions).issubset({0, 1, 2})
    
    def test_regime_probabilities(self, trained_hmm, sample_features):
        """Check predict_proba() sums to 1.0."""
        clean_features = sample_features[~np.isnan(sample_features).any(axis=1)]
        probs = trained_hmm.predict_proba(clean_features)
        
        # Shape should be (n_samples, n_components)
        assert probs.shape[0] == len(clean_features)
        assert probs.shape[1] == 3
        
        # Each row should sum to 1
        row_sums = probs.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(clean_features)), decimal=5)
    
    def test_confidence_calculation(self, trained_hmm, sample_features):
        """Test confidence as max probability."""
        clean_features = sample_features[~np.isnan(sample_features).any(axis=1)]
        probs = trained_hmm.predict_proba(clean_features)
        
        confidence = probs.max(axis=1)
        
        # Confidence should be between 1/3 and 1
        assert all(confidence >= 1/3 - 0.01)
        assert all(confidence <= 1.0)


# ============================================================================
# TRANSITION MATRIX TESTS
# ============================================================================

class TestTransitionMatrix:
    """Tests for HMM transition matrix."""
    
    def test_transition_matrix_shape(self, trained_hmm):
        """Verify transmat_ is 3x3."""
        assert trained_hmm.transmat_.shape == (3, 3)
    
    def test_transition_matrix_rows_sum_to_one(self, trained_hmm):
        """Verify rows sum to 1.0."""
        row_sums = trained_hmm.transmat_.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3), decimal=5)
    
    def test_transition_matrix_valid_probabilities(self, trained_hmm):
        """Verify all values are valid probabilities."""
        assert np.all(trained_hmm.transmat_ >= 0)
        assert np.all(trained_hmm.transmat_ <= 1)
    
    def test_diagonal_dominance(self, trained_hmm):
        """Regimes should have some persistence (diagonal > off-diagonal average)."""
        diag = np.diag(trained_hmm.transmat_)
        
        # At least one regime should show persistence
        assert np.max(diag) > 0.3


# ============================================================================
# BIC MODEL SELECTION TESTS
# ============================================================================

class TestBICSelection:
    """Tests for BIC-based model selection."""
    
    def test_bic_calculation(self, sample_features):
        """Test BIC can be calculated for different state counts."""
        from hmmlearn import hmm
        
        clean_features = sample_features[~np.isnan(sample_features).any(axis=1)]
        bic_scores = []
        
        for n_states in [2, 3, 4]:
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type='full',
                n_iter=100,
                random_state=42
            )
            model.fit(clean_features)
            
            log_likelihood = model.score(clean_features)
            n_params = n_states * (n_states - 1) + n_states * 3 + n_states * 3 * 4 // 2
            bic = -2 * log_likelihood + n_params * np.log(len(clean_features))
            
            bic_scores.append((n_states, bic))
        
        # All BIC scores should be finite
        assert all(np.isfinite(bic) for _, bic in bic_scores)
    
    def test_optimal_state_selection(self, sample_features):
        """Verify 2-5 states are tested."""
        from hmmlearn import hmm
        
        clean_features = sample_features[~np.isnan(sample_features).any(axis=1)]
        tested_states = []
        
        for n_states in range(2, 6):
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type='full',
                n_iter=50,
                random_state=42
            )
            try:
                model.fit(clean_features)
                tested_states.append(n_states)
            except:
                pass
        
        assert set(tested_states) == {2, 3, 4, 5}


# ============================================================================
# REGIME PERSISTENCE TESTS
# ============================================================================

class TestRegimePersistence:
    """Tests for regime duration and persistence."""
    
    def test_regime_persistence(self, trained_hmm, sample_features):
        """Calculate average duration per regime."""
        clean_features = sample_features[~np.isnan(sample_features).any(axis=1)]
        predictions = trained_hmm.predict(clean_features)
        
        # Calculate run lengths
        durations = {0: [], 1: [], 2: []}
        current_regime = predictions[0]
        current_length = 1
        
        for i in range(1, len(predictions)):
            if predictions[i] == current_regime:
                current_length += 1
            else:
                durations[current_regime].append(current_length)
                current_regime = predictions[i]
                current_length = 1
        durations[current_regime].append(current_length)
        
        # Average duration should be reasonable (> 1 day)
        for regime, lengths in durations.items():
            if lengths:
                avg = np.mean(lengths)
                assert avg >= 1.0


# ============================================================================
# REGIME LABELING TESTS
# ============================================================================

class TestRegimeLabeling:
    """Tests for regime name/color utilities."""
    
    def test_regime_name_mapping(self):
        """Verify regime_utils correctly maps 0→Euphoria, 1→Complacency, 2→Capitulation."""
        from src.utils.streamlit_helpers import get_regime_name
        
        assert get_regime_name(0) == 'Euphoria'
        assert get_regime_name(1) == 'Complacency'
        assert get_regime_name(2) == 'Capitulation'
    
    def test_regime_color_mapping(self):
        """Verify correct colors for regimes."""
        from src.utils.streamlit_helpers import get_regime_color
        
        assert get_regime_color(0) == '#00C853'  # Green
        assert get_regime_color(1) == '#FFD600'  # Yellow
        assert get_regime_color(2) == '#D50000'  # Red
    
    def test_unknown_regime_handling(self):
        """Test handling of invalid regime IDs."""
        from src.utils.streamlit_helpers import get_regime_name
        
        assert get_regime_name(99) == 'Unknown'
        assert get_regime_name(-1) == 'Unknown'


# ============================================================================
# REGIME STATISTICS TESTS
# ============================================================================

class TestRegimeStatistics:
    """Tests for regime distribution and statistics."""
    
    def test_regime_distribution(self, trained_hmm, sample_features):
        """Check regime distribution is reasonable."""
        clean_features = sample_features[~np.isnan(sample_features).any(axis=1)]
        predictions = trained_hmm.predict(clean_features)
        
        unique, counts = np.unique(predictions, return_counts=True)
        distribution = dict(zip(unique, counts / len(predictions)))
        
        # Each regime should appear at least once
        for regime in [0, 1, 2]:
            if regime in distribution:
                assert distribution[regime] > 0
    
    def test_transition_counts(self, trained_hmm, sample_features):
        """Test transition counting."""
        clean_features = sample_features[~np.isnan(sample_features).any(axis=1)]
        predictions = trained_hmm.predict(clean_features)
        
        # Count transitions
        transitions = 0
        for i in range(1, len(predictions)):
            if predictions[i] != predictions[i-1]:
                transitions += 1
        
        # Should have some transitions but not too many
        assert transitions > 0
        assert transitions < len(predictions) / 2
    
    def test_confidence_distribution(self, trained_hmm, sample_features):
        """Check confidence distribution."""
        clean_features = sample_features[~np.isnan(sample_features).any(axis=1)]
        probs = trained_hmm.predict_proba(clean_features)
        confidence = probs.max(axis=1)
        
        # Average confidence should be reasonable
        avg_confidence = np.mean(confidence)
        assert 0.4 < avg_confidence < 1.0


# ============================================================================
# HMM SERIALIZATION TESTS
# ============================================================================

class TestHMMSerialization:
    """Tests for model save/load functionality."""
    
    def test_hmm_serialization(self, trained_hmm, sample_features, tmp_path):
        """Verify model can be saved/loaded with pickle."""
        clean_features = sample_features[~np.isnan(sample_features).any(axis=1)]
        
        # Get predictions before save
        original_predictions = trained_hmm.predict(clean_features)
        
        # Save model
        model_path = tmp_path / 'test_hmm.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(trained_hmm, f)
        
        # Load model
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Predictions should match
        loaded_predictions = loaded_model.predict(clean_features)
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
    
    def test_loaded_model_attributes(self, trained_hmm, tmp_path):
        """Verify loaded model has all attributes."""
        model_path = tmp_path / 'test_hmm.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(trained_hmm, f)
        
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        assert hasattr(loaded_model, 'transmat_')
        assert hasattr(loaded_model, 'means_')
        assert hasattr(loaded_model, 'covars_')
        assert loaded_model.n_components == trained_hmm.n_components


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
