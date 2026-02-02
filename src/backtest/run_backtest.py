"""
CHRONOS Phase 5: Walk-Forward Backtesting CLI

This script runs the complete backtesting pipeline:
1. Loads trained models (HMM and XGBoost ensemble)
2. Loads processed features and returns data
3. Executes walk-forward backtest
4. Calculates performance metrics
5. Exports results to multiple formats
"""

import argparse
import logging
import sys
import os
from datetime import datetime

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from config import (
    INITIAL_CAPITAL,
    TEST_START,
    TEST_END,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR
)

from src.backtest.walk_forward import WalkForwardBacktest
from src.backtest.performance_tracker import BacktestPerformanceTracker
from src.backtest.results_exporter import ResultsExporter
from src.backtest.crisis_analyzer import CrisisAnalyzer
from src.models.regime_detector import RegimeDetector


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUTS_DIR, 'backtest', 'backtest.log'))
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    """Load features and returns data."""
    logger.info("Loading data...")
    
    features_path = os.path.join(PROCESSED_DATA_DIR, 'features.csv')
    returns_path = os.path.join(PROCESSED_DATA_DIR, 'returns.csv')
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not os.path.exists(returns_path):
        raise FileNotFoundError(f"Returns file not found: {returns_path}")
    
    features_df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    
    logger.info(f"[OK] Loaded features: {features_df.shape}")
    logger.info(f"[OK] Loaded returns: {returns_df.shape}")
    
    return features_df, returns_df


def load_models():
    """Load trained models."""
    logger.info("Loading trained models...")
    
    # Load regime detector
    regime_detector_path = os.path.join(MODELS_DIR, 'regime_detector.pkl')
    if not os.path.exists(regime_detector_path):
        raise FileNotFoundError(f"Regime detector not found: {regime_detector_path}")
    
    regime_detector = RegimeDetector.load_model(regime_detector_path)
    logger.info(f"[OK] Loaded RegimeDetector from {regime_detector_path}")
    
    # Load XGBoost ensemble
    import json
    import xgboost as xgb
    
    metadata_path = os.path.join(MODELS_DIR, 'ensemble_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    feature_names = metadata.get('feature_names', [])
    
    models = {}
    for regime in [0, 1, 2]:
        regime_names = {0: 'euphoria', 1: 'complacency', 2: 'capitulation'}
        model_path = os.path.join(MODELS_DIR, f'xgb_regime_{regime}_{regime_names[regime]}.json')
        
        if os.path.exists(model_path):
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            models[regime] = model
            logger.info(f"[OK] Loaded XGBoost model for regime {regime}")
        else:
            logger.warning(f"[WARN] Model not found for regime {regime}: {model_path}")
    
    ensemble = {
        'models': models,
        'feature_names': feature_names
    }
    
    logger.info(f"[OK] Loaded RegimeSpecializedEnsemble from {MODELS_DIR}")
    
    return regime_detector, ensemble


def run_backtest(start_date: str, end_date: str):
    """
    Run the complete backtest pipeline.
    
    Args:
        start_date: Backtest start date
        end_date: Backtest end date
    """
    print("=" * 60)
    print("CHRONOS PHASE 5: WALK-FORWARD BACKTESTING")
    print("=" * 60)
    print()
    
    # Load data and models
    features_df, returns_df = load_data()
    regime_detector, ensemble = load_models()
    
    print()
    print("Starting walk-forward backtest simulation...")
    print()
    
    # Initialize backtest engine
    backtest = WalkForwardBacktest(
        features_df=features_df,
        returns_df=returns_df,
        regime_detector=regime_detector,
        ensemble=ensemble,
        initial_capital=INITIAL_CAPITAL
    )
    
    # Run backtest
    results_df = backtest.run(start_date=start_date, end_date=end_date)
    
    # Get trade log and transitions
    trade_log = backtest.get_trade_log()
    regime_transitions = backtest.get_regime_transitions()
    
    print()
    print("=" * 60)
    print("CALCULATING PERFORMANCE METRICS")
    print("=" * 60)
    
    # Calculate performance metrics
    tracker = BacktestPerformanceTracker(results_df)
    summary_stats = tracker.calculate_summary_statistics()
    regime_performance = tracker.calculate_regime_performance()
    
    # Print summary
    print()
    print(f"Final Portfolio Value: ${summary_stats['final_value_chronos']:,.2f}")
    print(f"Final Benchmark Value: ${summary_stats['final_value_benchmark']:,.2f}")
    print(f"Total Return (CHRONOS): {summary_stats['total_return_chronos']:.2%}")
    print(f"Total Return (Benchmark): {summary_stats['total_return_benchmark']:.2%}")
    print(f"Outperformance: {summary_stats['outperformance']:.2%}")
    print(f"Sharpe Ratio: {summary_stats['sharpe_ratio_chronos']:.2f}")
    print(f"Max Drawdown: {summary_stats['max_dd_chronos']:.2%}")
    print(f"Win Rate: {summary_stats['win_rate']:.1%}")
    print()
    
    # Crisis analysis
    print("=" * 60)
    print("CRISIS PERIOD ANALYSIS")
    print("=" * 60)
    
    crisis_analyzer = CrisisAnalyzer(results_df)
    crisis_analysis = crisis_analyzer.analyze_all_crises()
    crisis_summary = crisis_analyzer.get_crisis_summary()
    
    print(f"Crises in test period: {crisis_summary['crises_in_test_period']}")
    print()
    
    # Export results
    print("=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)
    
    exporter = ResultsExporter()
    output_files = exporter.export_all(
        results_df=results_df,
        summary_stats=summary_stats,
        regime_performance=regime_performance,
        trade_log=trade_log,
        regime_transitions=regime_transitions,
        crisis_analysis=crisis_analysis
    )
    
    # Generate crisis report
    exporter.generate_crisis_report(crisis_analysis, summary_stats)
    
    print()
    for output_type, filepath in output_files.items():
        print(f"[OK] {output_type}: {os.path.basename(filepath)}")
    
    print()
    print("=" * 60)
    print("BACKTEST COMPLETE!")
    print("=" * 60)
    
    return results_df, summary_stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='CHRONOS Walk-Forward Backtesting')
    parser.add_argument('--start-date', type=str, default=TEST_START,
                        help=f'Backtest start date (default: {TEST_START})')
    parser.add_argument('--end-date', type=str, default=TEST_END,
                        help=f'Backtest end date (default: {TEST_END})')
    
    args = parser.parse_args()
    
    try:
        results_df, summary_stats = run_backtest(args.start_date, args.end_date)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
