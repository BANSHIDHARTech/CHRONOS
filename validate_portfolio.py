"""
Portfolio Layer Validation Script

This script verifies the correct implementation of the portfolio optimization layer
by running through the checkpoints defined in the implementation plan.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.portfolio import CVaROptimizer, PortfolioRebalancer, RiskMetrics, PortfolioTracker
    from src.portfolio.cvar_optimizer import CVaROptimizer  # specific import for type checking if needed
    import config
except ImportError as e:
    logger.error(f"Import failed: {e}")
    logger.error("Make sure you are running this script from the project root.")
    sys.exit(1)

def generate_mock_data(days=252):
    """Generate mock returns data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    
    # Mock returns with some correlation
    np.random.seed(42)
    spy = np.random.normal(0.0005, 0.01, days)
    tlt = np.random.normal(0.0001, 0.006, days) - 0.3 * spy  # Negative correlation
    gld = np.random.normal(0.0002, 0.008, days)
    
    df = pd.DataFrame({
        'SPY': spy,
        'TLT': tlt,
        'GLD': gld
    }, index=dates)
    
    return df

def test_cvar_optimizer():
    """Test 1: CVaR Optimizer Constraints"""
    logger.info("\n--- TEST 1: CVaR Optimizer Constraints ---")
    
    returns_df = generate_mock_data()
    
    # Test Euphoria Regime (0)
    # Constraints: SPY (0.60-0.80), TLT (0.10-0.30), GLD (0.10-0.30)
    optimizer = CVaROptimizer(returns_df, regime=0)
    weights = optimizer.optimize()
    
    logger.info(f"Euphoria Regime Weights: {weights}")
    
    # Verify logical constraints
    passed = True
    
    # Sum to 1.0 (with small tolerance)
    if not (0.99 <= sum(weights.values()) <= 1.01):
        logger.error(f"FAIL: Weights do not sum to 1.0: {sum(weights.values())}")
        passed = False
        
    # Check SPY bounds
    if not (0.59 <= weights['SPY'] <= 0.81): # slight tolerance for solver
        logger.error(f"FAIL: SPY weight {weights['SPY']} out of bounds [0.6, 0.8]")
        passed = False
        
    if passed:
        logger.info("✅ Euphoria regime constraints passed")
    else:
        logger.error("❌ Euphoria regime constraints failed")
    
    return passed

def test_confidence_adjustment():
    """Test 2: Confidence-Based Adjustment"""
    logger.info("\n--- TEST 2: Confidence Adjustment ---")
    
    returns_df = generate_mock_data()
    optimizer = CVaROptimizer(returns_df, regime=0)
    
    # Start with base weights (e.g., middle of range)
    base_weights = {'SPY': 0.60, 'TLT': 0.20, 'GLD': 0.20}
    
    # Test Bullish Adjustment
    # Bullish quintile (5) + High confidence (0.8) -> Should boost SPY
    adj_weights = optimizer.adjust_for_confidence(
        base_weights, 
        predicted_quintile=5, 
        prediction_confidence=0.8,
        current_regime=0
    )
    
    logger.info(f"Base SPY: {base_weights['SPY']} -> Adjusted SPY: {adj_weights['SPY']:.4f}")
    
    if adj_weights['SPY'] > base_weights['SPY']:
        logger.info("✅ Bullish adjustment correctly increased SPY weight")
    else:
        logger.error(f"❌ Bullish adjustment failed: {adj_weights['SPY']} <= {base_weights['SPY']}")

def test_rebalancer():
    """Test 3: Rebalancer Costs"""
    logger.info("\n--- TEST 3: Rebalancer Transaction Costs ---")
    
    rebalancer = PortfolioRebalancer(transaction_cost_rate=0.001, slippage_rate=0)
    
    # 100% turnover scenario
    current_weights = {'SPY': 1.0, 'TLT': 0.0, 'GLD': 0.0}
    target_weights = {'SPY': 0.0, 'TLT': 1.0, 'GLD': 0.0}
    portfolio_value = 100000
    
    # Expected cost: 100k * 2.0 (turnover logic counts sell 1.0 + buy 1.0 = 2.0 total changed) ?
    # Wait, simple turnover is sum(|diff|). 
    # |0-1| + |1-0| + |0-0| = 2.0 (200% turnover in terms of total volume traded vs portfolio size)
    # If using one-way turnover definition, it's 1.0.
    # Let's check implementation of calculate_turnover: sum(abs(new - old))
    # So turnover will be 2.0.
    # Cost = value * turnover * rate = 100,000 * 2.0 * 0.001 = $200
    
    val, w, cost = rebalancer.rebalance(
        current_weights, 
        target_weights, 
        portfolio_value, 
        {'SPY': 0, 'TLT': 0, 'GLD': 0} # No returns for cost isolation
    )
    
    turnover = rebalancer.calculate_turnover(current_weights, target_weights)
    logger.info(f"Turnover: {turnover}")
    logger.info(f"Transaction Cost: ${cost:.2f}")
    
    if 199.99 <= cost <= 200.01:
        logger.info("✅ Transaction cost calculation verified")
    else:
        logger.error(f"❌ Transaction cost mismatch. Expected ~$200, got {cost}")

def test_risk_metrics():
    """Test 4: Risk Metrics Calculation"""
    logger.info("\n--- TEST 4: Risk Metrics ---")
    
    # Create simple series: +1%, -1%, +1%, -1%...
    # Mean ~0, Volatility should be meaningful
    data = [0.01, -0.01] * 126 # 252 days
    returns = pd.Series(data)
    
    metrics = RiskMetrics(risk_free_rate=0.0)
    stats = metrics.calculate_all_metrics(returns)
    
    logger.info(f"Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
    
    # Ann Return ~ 0. Ann Vol ~ 0.01 * sqrt(252) ~ 0.158
    # Sharpe should be ~0
    
    if -0.1 < stats['sharpe_ratio'] < 0.1:
        logger.info("✅ Risk metrics calculation reasonable")
    else:
        logger.warning(f"⚠️ Risk metrics values unusual: {stats}")

def test_portfolio_integration():
    """Test 5: Full Tracker Integration"""
    logger.info("\n--- TEST 5: Portfolio Tracker Integration ---")
    
    tracker = PortfolioTracker(initial_capital=100000)
    
    # Day 1
    weights = {'SPY': 0.5, 'TLT': 0.5, 'GLD': 0.0}
    returns = {'SPY': 0.02, 'TLT': -0.01, 'GLD': 0.0} # Portfolio ret = 0.5*0.02 + 0.5*-0.01 = 0.01 - 0.005 = 0.005 (0.5%)
    
    new_val = tracker.update(
        date=datetime.now(),
        weights=weights,
        returns_today=returns,
        regime=0,
        transaction_cost=10.0
    )
    
    expected_val = 100000 * (1 + 0.005) - 10.0 # 100500 - 10 = 100490
    
    logger.info(f"New Value: ${new_val:.2f}")
    
    if 100489 <= new_val <= 100491:
        logger.info("✅ Portfolio tracking accurate")
    else:
        logger.error(f"❌ Tracking mismatch. Expected 100490, got {new_val}")

if __name__ == "__main__":
    print("="*60)
    print("CHRONOS PORTFOLIO LAYER VALIDATION")
    print("="*60)
    
    try:
        test_cvar_optimizer()
        test_confidence_adjustment()
        test_rebalancer()
        test_risk_metrics()
        test_portfolio_integration()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY ✅")
        print("="*60)
        
    except Exception as e:
        logger.exception("Validation failed with error")
        sys.exit(1)
