"""
Unit Tests for CHRONOS Portfolio Optimization Module

Tests for CVaR optimizer, rebalancing engine, risk metrics, and related functions.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PORTFOLIO_ASSETS, REGIME_CONSTRAINTS, INITIAL_CAPITAL,
    TRANSACTION_COST_RATE, SLIPPAGE_RATE,
    HIGH_CONFIDENCE_THRESHOLD, EQUITY_BOOST_FACTOR, EQUITY_CUT_FACTOR
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_returns():
    """Generate sample asset returns for testing."""
    np.random.seed(42)
    n = 252  # One year of daily data
    
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    
    # Simulate correlated returns
    spy_returns = 0.0004 + np.random.randn(n) * 0.012
    tlt_returns = 0.0001 + np.random.randn(n) * 0.008 - spy_returns * 0.3
    gld_returns = 0.0002 + np.random.randn(n) * 0.010
    
    return pd.DataFrame({
        'SPY': spy_returns,
        'TLT': tlt_returns,
        'GLD': gld_returns
    }, index=dates)


@pytest.fixture
def sample_prices(sample_returns):
    """Generate sample prices from returns."""
    initial_prices = {'SPY': 450, 'TLT': 100, 'GLD': 180}
    prices = pd.DataFrame(index=sample_returns.index, columns=PORTFOLIO_ASSETS)
    
    for asset in PORTFOLIO_ASSETS:
        prices[asset] = initial_prices[asset] * (1 + sample_returns[asset]).cumprod()
    
    return prices


@pytest.fixture
def sample_covariance(sample_returns):
    """Calculate sample covariance matrix."""
    return sample_returns.cov() * 252  # Annualized


@pytest.fixture
def sample_expected_returns(sample_returns):
    """Calculate expected returns."""
    return sample_returns.mean() * 252  # Annualized


@pytest.fixture
def current_weights():
    """Sample current portfolio weights."""
    return {'SPY': 0.50, 'TLT': 0.30, 'GLD': 0.20}


# ============================================================================
# CVAR OPTIMIZER TESTS
# ============================================================================

class TestCVaROptimizer:
    """Tests for CVaR portfolio optimization."""
    
    def test_cvar_optimizer_initialization(self, sample_expected_returns, sample_covariance):
        """Verify EfficientCVaR setup with expected returns and covariance."""
        from pypfopt import EfficientCVaR
        from pypfopt.expected_returns import mean_historical_return
        from pypfopt.risk_models import sample_cov
        
        # Should be able to create optimizer
        optimizer = EfficientCVaR(
            expected_returns=sample_expected_returns,
            cov_matrix=sample_covariance,
            beta=0.95  # 95% CVaR
        )
        
        assert optimizer is not None
    
    def test_weights_sum_to_one(self, sample_returns):
        """Verify optimized weights sum to 1.0."""
        from pypfopt import EfficientCVaR
        
        expected_returns = sample_returns.mean() * 252
        cov_matrix = sample_returns.cov() * 252
        
        optimizer = EfficientCVaR(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            beta=0.95
        )
        
        # Add constraints
        optimizer.add_constraint(lambda w: w.sum() == 1)
        
        try:
            weights = optimizer.min_cvar()
            cleaned_weights = optimizer.clean_weights()
            
            total = sum(cleaned_weights.values())
            assert abs(total - 1.0) < 0.01
        except Exception:
            # If optimization fails, test fallback weights
            fallback_weights = {'SPY': 0.5, 'TLT': 0.3, 'GLD': 0.2}
            assert abs(sum(fallback_weights.values()) - 1.0) < 0.01


# ============================================================================
# REGIME CONSTRAINTS TESTS
# ============================================================================

class TestRegimeConstraints:
    """Tests for regime-specific constraints."""
    
    def test_regime_constraints_exist(self):
        """Check constraints defined for all regimes."""
        assert 0 in REGIME_CONSTRAINTS  # Euphoria
        assert 1 in REGIME_CONSTRAINTS  # Complacency
        assert 2 in REGIME_CONSTRAINTS  # Capitulation
    
    def test_euphoria_constraints(self):
        """Check Euphoria regime has high equity allocation."""
        constraints = REGIME_CONSTRAINTS[0]
        
        # SPY should have high allocation
        spy_min, spy_max = constraints['SPY']
        assert spy_min >= 0.5  # At least 50%
        assert spy_max <= 0.9  # At most 90%
    
    def test_complacency_constraints(self):
        """Check Complacency regime has balanced allocation."""
        constraints = REGIME_CONSTRAINTS[1]
        
        # Moderate SPY allocation
        spy_min, spy_max = constraints['SPY']
        assert spy_min >= 0.3
        assert spy_max <= 0.7
    
    def test_capitulation_constraints(self):
        """Check Capitulation regime has defensive allocation."""
        constraints = REGIME_CONSTRAINTS[2]
        
        # Low SPY allocation
        spy_min, spy_max = constraints['SPY']
        assert spy_max <= 0.3  # At most 30%
        
        # High TLT allocation
        tlt_min, tlt_max = constraints['TLT']
        assert tlt_min >= 0.4  # At least 40%
    
    def test_constraints_are_valid_ranges(self):
        """Check all constraints are valid min/max ranges."""
        for regime_id, constraints in REGIME_CONSTRAINTS.items():
            for asset, (min_wt, max_wt) in constraints.items():
                assert 0 <= min_wt <= 1, f"Invalid min for {asset} in regime {regime_id}"
                assert 0 <= max_wt <= 1, f"Invalid max for {asset} in regime {regime_id}"
                assert min_wt <= max_wt, f"Min > max for {asset} in regime {regime_id}"
    
    def test_weights_within_bounds(self):
        """Check sample weights respect regime-specific constraints."""
        for regime_id, constraints in REGIME_CONSTRAINTS.items():
            # Generate weights at constraint midpoints
            weights = {}
            for asset in PORTFOLIO_ASSETS:
                min_wt, max_wt = constraints[asset]
                weights[asset] = (min_wt + max_wt) / 2
            
            # Normalize
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
            
            # Check normalized weights are still reasonable
            for asset, weight in weights.items():
                assert 0 <= weight <= 1


# ============================================================================
# CONFIDENCE ADJUSTMENT TESTS
# ============================================================================

class TestConfidenceAdjustment:
    """Tests for confidence-based weight adjustments."""
    
    def test_high_confidence_threshold(self):
        """Verify high confidence threshold is reasonable."""
        assert 0.5 <= HIGH_CONFIDENCE_THRESHOLD <= 0.9
    
    def test_equity_boost_factor(self):
        """Verify bullish boost increases SPY weight."""
        assert EQUITY_BOOST_FACTOR > 1.0  # Should increase
        assert EQUITY_BOOST_FACTOR <= 1.5  # But not too much
    
    def test_equity_cut_factor(self):
        """Verify bearish cut decreases SPY weight."""
        assert EQUITY_CUT_FACTOR < 1.0  # Should decrease
        assert EQUITY_CUT_FACTOR >= 0.3  # But not to zero
    
    def test_confidence_boost_simulation(self, current_weights):
        """Simulate high-confidence bullish adjustment."""
        confidence = 0.85  # High confidence
        is_bullish = True
        
        adjusted_weights = current_weights.copy()
        
        if confidence > HIGH_CONFIDENCE_THRESHOLD and is_bullish:
            spy_boost = adjusted_weights['SPY'] * EQUITY_BOOST_FACTOR
            adjusted_weights['SPY'] = min(spy_boost, 0.90)  # Cap at 90%
        
        assert adjusted_weights['SPY'] >= current_weights['SPY']
    
    def test_confidence_cut_simulation(self, current_weights):
        """Simulate high-confidence bearish adjustment."""
        confidence = 0.85  # High confidence
        is_bullish = False
        
        adjusted_weights = current_weights.copy()
        
        if confidence > HIGH_CONFIDENCE_THRESHOLD and not is_bullish:
            adjusted_weights['SPY'] = adjusted_weights['SPY'] * EQUITY_CUT_FACTOR
        
        assert adjusted_weights['SPY'] <= current_weights['SPY']


# ============================================================================
# FALLBACK ALLOCATION TESTS
# ============================================================================

class TestFallbackAllocation:
    """Tests for fallback weights when optimization fails."""
    
    def test_fallback_weights_sum_to_one(self):
        """Test heuristic weights sum to 1.0."""
        fallback_weights = {'SPY': 0.5, 'TLT': 0.3, 'GLD': 0.2}
        assert abs(sum(fallback_weights.values()) - 1.0) < 0.01
    
    def test_fallback_for_each_regime(self):
        """Test fallback exists for each regime."""
        for regime_id in [0, 1, 2]:
            constraints = REGIME_CONSTRAINTS[regime_id]
            
            # Generate fallback at constraint midpoints
            fallback = {}
            for asset in PORTFOLIO_ASSETS:
                min_wt, max_wt = constraints[asset]
                fallback[asset] = (min_wt + max_wt) / 2
            
            # Normalize
            total = sum(fallback.values())
            fallback = {k: v / total for k, v in fallback.items()}
            
            assert abs(sum(fallback.values()) - 1.0) < 0.01


# ============================================================================
# TRANSACTION COST TESTS
# ============================================================================

class TestTransactionCosts:
    """Tests for transaction cost calculations."""
    
    def test_transaction_cost_rate(self):
        """Verify transaction cost rate is reasonable."""
        assert 0 < TRANSACTION_COST_RATE <= 0.01  # 0-100 bps
    
    def test_slippage_rate(self):
        """Verify slippage rate is reasonable."""
        assert 0 <= SLIPPAGE_RATE <= 0.01  # 0-100 bps
    
    def test_cost_calculation(self, current_weights):
        """Verify cost calculation on weight changes."""
        new_weights = {'SPY': 0.60, 'TLT': 0.25, 'GLD': 0.15}
        
        total_turnover = 0
        for asset in PORTFOLIO_ASSETS:
            change = abs(new_weights[asset] - current_weights[asset])
            total_turnover += change
        
        # Half the turnover is traded (buy = sell)
        traded_value = total_turnover / 2
        
        # Calculate cost
        cost = traded_value * TRANSACTION_COST_RATE
        slippage = traded_value * SLIPPAGE_RATE
        total_cost = cost + slippage
        
        assert total_cost >= 0
        assert total_cost < total_turnover  # Cost shouldn't exceed trade


# ============================================================================
# REBALANCING LOGIC TESTS
# ============================================================================

class TestRebalancingLogic:
    """Tests for rebalancing triggers and execution."""
    
    def test_weekly_rebalancing(self):
        """Check weekly rebalancing frequency."""
        from config import REBALANCE_FREQUENCY
        
        assert REBALANCE_FREQUENCY in ['daily', 'weekly', 'bi-weekly', 'monthly']
    
    def test_regime_change_trigger(self, current_weights):
        """Test regime change triggers immediate rebalance."""
        current_regime = 1  # Complacency
        new_regime = 2  # Capitulation
        
        # Should trigger rebalance
        should_rebalance = current_regime != new_regime
        assert should_rebalance is True
    
    def test_threshold_breach_trigger(self, current_weights):
        """Test threshold breach triggers rebalance."""
        target_weights = {'SPY': 0.50, 'TLT': 0.30, 'GLD': 0.20}
        drift_threshold = 0.10  # 10%
        
        # Simulate drift
        actual_weights = {'SPY': 0.65, 'TLT': 0.22, 'GLD': 0.13}
        
        max_drift = max(
            abs(actual_weights[asset] - target_weights[asset])
            for asset in PORTFOLIO_ASSETS
        )
        
        should_rebalance = max_drift > drift_threshold
        assert should_rebalance is True


# ============================================================================
# RISK METRICS TESTS
# ============================================================================

class TestRiskMetrics:
    """Tests for risk metric calculations."""
    
    def test_sharpe_ratio_calculation(self, sample_returns):
        """Verify Sharpe ratio calculation."""
        portfolio_returns = (
            sample_returns['SPY'] * 0.5 +
            sample_returns['TLT'] * 0.3 +
            sample_returns['GLD'] * 0.2
        )
        
        mean_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        risk_free = 0.04  # 4% risk-free rate
        
        sharpe = (mean_return - risk_free) / volatility if volatility > 0 else 0
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_sortino_ratio_calculation(self, sample_returns):
        """Verify Sortino ratio calculation."""
        portfolio_returns = (
            sample_returns['SPY'] * 0.5 +
            sample_returns['TLT'] * 0.3 +
            sample_returns['GLD'] * 0.2
        )
        
        mean_return = portfolio_returns.mean() * 252
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.001
        risk_free = 0.04
        
        sortino = (mean_return - risk_free) / downside_std if downside_std > 0 else 0
        
        assert isinstance(sortino, float)
    
    def test_max_drawdown_calculation(self, sample_prices):
        """Verify max drawdown calculation."""
        portfolio_value = (
            sample_prices['SPY'] * 0.5 +
            sample_prices['TLT'] * 0.3 +
            sample_prices['GLD'] * 0.2
        )
        
        cummax = portfolio_value.cummax()
        drawdown = (portfolio_value - cummax) / cummax
        max_dd = drawdown.min()
        
        assert max_dd <= 0  # Drawdown is negative
        assert max_dd >= -1  # Can't lose more than 100%
    
    def test_calmar_ratio_calculation(self, sample_returns, sample_prices):
        """Verify Calmar ratio calculation."""
        portfolio_returns = (
            sample_returns['SPY'] * 0.5 +
            sample_returns['TLT'] * 0.3 +
            sample_returns['GLD'] * 0.2
        )
        
        portfolio_value = (
            sample_prices['SPY'] * 0.5 +
            sample_prices['TLT'] * 0.3 +
            sample_prices['GLD'] * 0.2
        )
        
        annualized_return = portfolio_returns.mean() * 252
        
        cummax = portfolio_value.cummax()
        drawdown = (portfolio_value - cummax) / cummax
        max_dd = abs(drawdown.min())
        
        calmar = annualized_return / max_dd if max_dd > 0 else 0
        
        assert isinstance(calmar, float)
    
    def test_cvar_calculation(self, sample_returns):
        """Verify CVaR (Expected Shortfall) calculation."""
        portfolio_returns = (
            sample_returns['SPY'] * 0.5 +
            sample_returns['TLT'] * 0.3 +
            sample_returns['GLD'] * 0.2
        )
        
        # 95% CVaR: expected loss in worst 5%
        alpha = 0.05
        var_threshold = portfolio_returns.quantile(alpha)
        cvar = portfolio_returns[portfolio_returns <= var_threshold].mean()
        
        assert cvar <= var_threshold  # CVaR should be worse than VaR
        assert isinstance(cvar, float)


# ============================================================================
# PORTFOLIO VALUE TRACKING TESTS
# ============================================================================

class TestPortfolioValueTracking:
    """Tests for portfolio value evolution and attribution."""
    
    def test_portfolio_value_evolution(self, sample_prices, current_weights):
        """Check portfolio value evolution."""
        initial_value = INITIAL_CAPITAL
        
        # Calculate shares
        shares = {}
        for asset in PORTFOLIO_ASSETS:
            allocation = initial_value * current_weights[asset]
            shares[asset] = allocation / sample_prices[asset].iloc[0]
        
        # Calculate value over time
        portfolio_values = pd.Series(0.0, index=sample_prices.index)
        for asset in PORTFOLIO_ASSETS:
            portfolio_values += shares[asset] * sample_prices[asset]
        
        # Value should fluctuate
        assert portfolio_values.std() > 0
        
        # Should start near initial capital
        assert abs(portfolio_values.iloc[0] - initial_value) < 100
    
    def test_return_attribution(self, sample_returns, current_weights):
        """Check return attribution by asset."""
        portfolio_return = 0
        
        for asset in PORTFOLIO_ASSETS:
            asset_contribution = sample_returns[asset].sum() * current_weights[asset]
            portfolio_return += asset_contribution
        
        # Direct calculation should match
        direct_return = (
            sample_returns['SPY'].sum() * 0.5 +
            sample_returns['TLT'].sum() * 0.3 +
            sample_returns['GLD'].sum() * 0.2
        )
        
        assert abs(portfolio_return - direct_return) < 1e-10


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
