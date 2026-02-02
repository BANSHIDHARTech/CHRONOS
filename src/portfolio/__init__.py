"""
Portfolio Optimization Module

This module provides institutional-grade portfolio optimization and management
functionality including:
- CVaR (Conditional Value at Risk) optimization with regime-specific constraints
- Portfolio rebalancing with transaction cost modeling
- Comprehensive risk metrics calculation
- Portfolio value and attribution tracking

Example:
    from src.portfolio import CVaROptimizer, PortfolioRebalancer, RiskMetrics, PortfolioTracker
    
    # Initialize optimizer with historical returns and current regime
    optimizer = CVaROptimizer(returns_df, regime=1)
    weights = optimizer.optimize()
    
    # Adjust weights based on prediction confidence
    adjusted_weights = optimizer.adjust_for_confidence(
        weights, predicted_quintile=5, prediction_confidence=0.75, current_regime=1
    )
    
    # Calculate risk metrics
    metrics = RiskMetrics()
    risk_stats = metrics.calculate_all_metrics(portfolio_returns)
"""

from .cvar_optimizer import CVaROptimizer
from .rebalancer import PortfolioRebalancer
from .risk_metrics import RiskMetrics
from .portfolio_tracker import PortfolioTracker

__all__ = [
    'CVaROptimizer',
    'PortfolioRebalancer',
    'RiskMetrics',
    'PortfolioTracker',
]
