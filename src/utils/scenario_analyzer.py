"""
What-If Scenario Analyzer for CHRONOS Dashboard

Provides functionality to analyze regime override scenarios
and compare portfolio allocations under different conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import (
    REGIME_CONSTRAINTS, PORTFOLIO_ASSETS,
    HIGH_CONFIDENCE_THRESHOLD, EQUITY_BOOST_FACTOR, EQUITY_CUT_FACTOR
)


def get_regime_allocation(regime_id: int) -> Dict[str, float]:
    """
    Get the target portfolio allocation for a given regime.
    
    Args:
        regime_id: Regime identifier (0, 1, or 2)
        
    Returns:
        Dictionary mapping asset names to target weights
    """
    constraints = REGIME_CONSTRAINTS.get(regime_id, REGIME_CONSTRAINTS[1])
    
    # Use midpoint of constraints as target
    allocation = {}
    for asset in PORTFOLIO_ASSETS:
        min_wt, max_wt = constraints[asset]
        allocation[asset] = (min_wt + max_wt) / 2
    
    # Normalize to sum to 1
    total = sum(allocation.values())
    if total > 0:
        allocation = {k: v / total for k, v in allocation.items()}
    
    return allocation


def analyze_regime_override(
    current_regime: int,
    override_regime: int,
    current_weights: Dict[str, float],
    confidence: float = 0.8
) -> Dict[str, Any]:
    """
    Analyze the impact of overriding the current regime.
    
    Args:
        current_regime: Current detected regime
        override_regime: User-specified override regime
        current_weights: Current portfolio weights
        confidence: Confidence level for the analysis
        
    Returns:
        Dictionary containing:
        - current_allocation: Current weights
        - override_allocation: New weights under override
        - weight_changes: Changes in weights
        - risk_impact: Estimated risk impact
    """
    # Get allocations for both regimes
    current_allocation = get_regime_allocation(current_regime)
    override_allocation = get_regime_allocation(override_regime)
    
    # Calculate weight changes
    weight_changes = {
        asset: override_allocation[asset] - current_allocation[asset]
        for asset in PORTFOLIO_ASSETS
    }
    
    # Estimate risk impact
    risk_impact = _estimate_risk_impact(current_regime, override_regime, confidence)
    
    # Calculate rebalancing requirements
    rebalance_required = _calculate_rebalance_needed(current_weights, override_allocation)
    
    return {
        'current_regime': current_regime,
        'override_regime': override_regime,
        'current_allocation': current_allocation,
        'override_allocation': override_allocation,
        'weight_changes': weight_changes,
        'risk_impact': risk_impact,
        'rebalance_required': rebalance_required,
        'confidence': confidence
    }


def _estimate_risk_impact(current_regime: int, override_regime: int, confidence: float) -> Dict[str, Any]:
    """
    Estimate the risk impact of switching regimes.
    
    Args:
        current_regime: Current regime
        override_regime: Override regime
        confidence: Confidence level
        
    Returns:
        Dictionary with risk metrics
    """
    # Risk characteristics by regime
    regime_risk = {
        0: {'volatility': 'High', 'expected_return': 'High', 'equity_exposure': 'Maximum'},
        1: {'volatility': 'Medium', 'expected_return': 'Medium', 'equity_exposure': 'Moderate'},
        2: {'volatility': 'Low', 'expected_return': 'Low', 'equity_exposure': 'Minimum'}
    }
    
    current_risk = regime_risk.get(current_regime, regime_risk[1])
    override_risk = regime_risk.get(override_regime, regime_risk[1])
    
    # Calculate risk direction
    risk_levels = {'High': 3, 'Medium': 2, 'Low': 1, 'Maximum': 3, 'Moderate': 2, 'Minimum': 1}
    
    current_level = risk_levels.get(current_risk['volatility'], 2)
    override_level = risk_levels.get(override_risk['volatility'], 2)
    
    if override_level > current_level:
        risk_direction = 'Increasing'
        risk_color = 'red'
    elif override_level < current_level:
        risk_direction = 'Decreasing'
        risk_color = 'green'
    else:
        risk_direction = 'Unchanged'
        risk_color = 'yellow'
    
    return {
        'current_risk_profile': current_risk,
        'override_risk_profile': override_risk,
        'risk_direction': risk_direction,
        'risk_color': risk_color,
        'adjustment_confidence': confidence
    }


def _calculate_rebalance_needed(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float]
) -> Dict[str, Any]:
    """
    Calculate the rebalancing required to reach target weights.
    
    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        
    Returns:
        Dictionary with rebalancing details
    """
    trades = {}
    total_turnover = 0
    
    for asset in PORTFOLIO_ASSETS:
        current = current_weights.get(asset, 0)
        target = target_weights.get(asset, 0)
        change = target - current
        
        if abs(change) > 0.01:  # 1% threshold
            action = 'BUY' if change > 0 else 'SELL'
            trades[asset] = {
                'action': action,
                'weight_change': change,
                'current_weight': current,
                'target_weight': target
            }
            total_turnover += abs(change)
    
    return {
        'trades': trades,
        'total_turnover': total_turnover,
        'estimated_cost': total_turnover * 0.001,  # 10bps transaction cost
        'rebalancing_needed': total_turnover > 0.05  # 5% threshold
    }


def get_scenario_comparison_table(analysis_result: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a comparison table for current vs override scenario.
    
    Args:
        analysis_result: Result from analyze_regime_override
        
    Returns:
        DataFrame with comparison data
    """
    rows = []
    
    for asset in PORTFOLIO_ASSETS:
        current = analysis_result['current_allocation'][asset]
        override = analysis_result['override_allocation'][asset]
        change = analysis_result['weight_changes'][asset]
        
        rows.append({
            'Asset': asset,
            'Current Weight': f"{current:.1%}",
            'Override Weight': f"{override:.1%}",
            'Change': f"{change:+.1%}",
            'Action': 'BUY' if change > 0 else ('SELL' if change < 0 else '-')
        })
    
    return pd.DataFrame(rows)


def simulate_scenario_performance(
    returns_df: pd.DataFrame,
    current_weights: Dict[str, float],
    override_weights: Dict[str, float],
    lookback_days: int = 252
) -> Dict[str, Any]:
    """
    Simulate historical performance under different weight scenarios.
    
    Args:
        returns_df: DataFrame with asset returns
        current_weights: Current portfolio weights
        override_weights: Override portfolio weights
        lookback_days: Number of days to simulate
        
    Returns:
        Dictionary with simulated performance metrics
    """
    # Get recent returns
    recent_returns = returns_df.tail(lookback_days)
    
    if len(recent_returns) < 20:
        return {
            'error': 'Insufficient data for simulation',
            'min_required': 20,
            'available': len(recent_returns)
        }
    
    # Calculate portfolio returns for each scenario
    current_portfolio = pd.Series(0.0, index=recent_returns.index)
    override_portfolio = pd.Series(0.0, index=recent_returns.index)
    
    for asset in PORTFOLIO_ASSETS:
        if asset in recent_returns.columns:
            current_portfolio += recent_returns[asset] * current_weights.get(asset, 0)
            override_portfolio += recent_returns[asset] * override_weights.get(asset, 0)
    
    # Calculate metrics
    def calc_metrics(returns):
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }
    
    current_metrics = calc_metrics(current_portfolio)
    override_metrics = calc_metrics(override_portfolio)
    
    # Calculate differences
    differences = {
        key: override_metrics[key] - current_metrics[key]
        for key in current_metrics
    }
    
    return {
        'current_scenario': current_metrics,
        'override_scenario': override_metrics,
        'differences': differences,
        'simulation_period': f"{recent_returns.index[0].date()} to {recent_returns.index[-1].date()}",
        'days_simulated': len(recent_returns)
    }


def get_regime_recommendation(
    current_regime: int,
    model_confidence: float,
    confidence_threshold: float = None,
    recent_volatility: float = None
) -> Dict[str, Any]:
    """
    Get a recommendation on whether to override the current regime.
    
    Args:
        current_regime: Current detected regime
        model_confidence: Model's confidence in the regime prediction
        confidence_threshold: User-specified threshold for high confidence (uses config default if None)
        recent_volatility: Recent market volatility (optional)
        
    Returns:
        Dictionary with recommendation details
    """
    regime_names = {0: 'Euphoria', 1: 'Complacency', 2: 'Capitulation'}
    
    # Use provided threshold or fall back to config default
    threshold = confidence_threshold if confidence_threshold is not None else HIGH_CONFIDENCE_THRESHOLD
    
    recommendation = {
        'current_regime': regime_names[current_regime],
        'model_confidence': model_confidence,
        'confidence_threshold': threshold,
        'override_suggested': False,
        'reason': '',
        'confidence_status': 'unknown'
    }
    
    # Low confidence suggests potential for override
    # Use threshold * 0.5 as "low" and threshold as "high"
    low_threshold = threshold * 0.5
    
    if model_confidence < low_threshold:
        recommendation['override_suggested'] = True
        recommendation['reason'] = f'Low model confidence (below {low_threshold:.0%}) - consider manual assessment'
        recommendation['suggested_action'] = 'Review recent market conditions manually'
        recommendation['confidence_status'] = 'low'
    
    elif model_confidence < threshold:
        recommendation['reason'] = f'Moderate confidence (below {threshold:.0%}) - regime detection may be in transition'
        recommendation['suggested_action'] = 'Monitor closely for regime change signals'
        recommendation['confidence_status'] = 'moderate'
    
    else:
        recommendation['reason'] = f'High confidence (above {threshold:.0%}) in regime detection'
        recommendation['suggested_action'] = 'Follow model allocation recommendations'
        recommendation['confidence_status'] = 'high'
    
    # Add volatility context if provided
    if recent_volatility is not None:
        if recent_volatility > 0.25:  # High volatility
            recommendation['volatility_note'] = 'High market volatility - consider more defensive positioning'
        elif recent_volatility < 0.10:  # Low volatility
            recommendation['volatility_note'] = 'Low market volatility - typical of complacent markets'
    
    return recommendation

