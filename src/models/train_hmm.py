"""
HMM Training Pipeline

End-to-end training script for the regime detector.
Loads data from Phase 1, engineers features, trains HMM, and saves results.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data.data_loader import DataLoader
from src.data.feature_engineering import FeatureEngineer
from src.models.regime_detector import RegimeDetector
from src.models.hmm_optimizer import select_optimal_states, plot_bic_curve
from src.utils.regime_utils import (
    export_regime_analysis,
    get_regime_summary_statistics,
    label_regimes,
    REGIME_NAMES
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data() -> pd.DataFrame:
    """
    Load data using DataLoader from Phase 1.
    
    Returns:
        Raw OHLCV data DataFrame
    """
    logger.info("Loading market data...")
    
    loader = DataLoader(raw_data_dir=config.RAW_DATA_DIR)
    
    tickers = list(config.TICKERS.keys())
    data = loader.get_data(
        tickers=tickers,
        start_date=config.DATA_START_DATE,
        end_date=config.DATA_END_DATE,
        use_cache=True
    )
    
    logger.info(f"Loaded data shape: {data.shape}")
    logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    return data


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features using FeatureEngineer from Phase 1.
    
    Args:
        data: Raw OHLCV data
        
    Returns:
        Engineered features DataFrame
    """
    logger.info("Engineering features...")
    
    engineer = FeatureEngineer(config)
    features = engineer.engineer_features(data, primary_ticker='SPY')
    
    logger.info(f"Engineered features shape: {features.shape}")
    logger.info(f"Features: {list(features.columns)}")
    
    return features


def extract_training_period(
    features: pd.DataFrame,
    train_start: str,
    train_end: str
) -> pd.DataFrame:
    """
    Extract training period (2019-2022) from features.
    
    Args:
        features: Full feature DataFrame
        train_start: Training start date
        train_end: Training end date
        
    Returns:
        Training period features
    """
    logger.info(f"Extracting training period: {train_start} to {train_end}")
    
    train_features = features.loc[train_start:train_end]
    
    logger.info(f"Training features shape: {train_features.shape}")
    logger.info(f"Training date range: {train_features.index[0]} to {train_features.index[-1]}")
    
    return train_features


def validate_training_data(X_train: np.ndarray, regimes: np.ndarray) -> bool:
    """
    Validate that training data and predictions are reasonable.
    
    Args:
        X_train: Training feature matrix
        regimes: Predicted regimes on training data
        
    Returns:
        True if validation passes
    """
    logger.info("Validating training results...")
    
    issues = []
    
    # Check all 3 regimes appear
    unique_regimes = np.unique(regimes)
    if len(unique_regimes) < 3:
        issues.append(f"Only {len(unique_regimes)} regimes detected (expected 3)")
    
    # Check regime distribution
    for regime in range(3):
        regime_pct = (regimes == regime).mean() * 100
        if regime_pct < 10:
            issues.append(f"Regime {REGIME_NAMES[regime]} is only {regime_pct:.1f}% of data (expected >10%)")
    
    # Check for degenerate regimes
    regime_counts = pd.Series(regimes).value_counts()
    logger.info(f"Regime distribution: {dict(regime_counts)}")
    
    if issues:
        for issue in issues:
            logger.warning(f"Validation issue: {issue}")
        return False
    
    logger.info("✓ Training validation passed")
    return True


def train_regime_detector(
    features: Optional[pd.DataFrame] = None,
    run_bic_optimization: bool = False,
    save_model: bool = True
) -> Dict[str, Any]:
    """
    Main training function for regime detector.
    
    Args:
        features: Pre-computed features (optional, will load if None)
        run_bic_optimization: Whether to run BIC optimization (default False)
        save_model: Whether to save the trained model
        
    Returns:
        Dictionary with training results
    """
    logger.info("="*60)
    logger.info("STARTING HMM REGIME DETECTOR TRAINING")
    logger.info("="*60)
    
    # Load and prepare data if not provided
    if features is None:
        data = load_and_prepare_data()
        features = engineer_features(data)
    
    # Extract training period
    train_features = extract_training_period(
        features,
        train_start=config.TRAIN_START,
        train_end=config.TRAIN_END
    )
    
    # Select HMM features - use 8 features for ROBUST regime separation
    # The key insight: Capitulation needs drawdown + momentum + fear metrics
    hmm_feature_cols = [
        'log_returns',      # Direction
        'parkinson_vol',    # Volatility
        'rolling_sharpe',   # Risk-adjusted
        'rolling_max_dd',   # CRITICAL: Drawdown for crisis detection
        'vix_level',        # Absolute fear
        'vix_percentile',   # Relative fear vs recent history
        'rsi',              # Momentum exhaustion (oversold/overbought)
        'momentum_20d'      # Trend direction
    ]
    
    # Verify columns exist
    missing_cols = [col for col in hmm_feature_cols if col not in train_features.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    X_train_raw = train_features[hmm_feature_cols].dropna()
    logger.info(f"HMM training features shape: {X_train_raw.shape}")
    
    # CRITICAL: Standardize features for HMM
    # Raw returns are ~0.0004 to -0.003 - too small for HMM to find distinct clusters
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw.values)
    logger.info("Features standardized with StandardScaler")
    logger.info(f"Feature means after scaling: {X_train_scaled.mean(axis=0)}")
    logger.info(f"Feature stds after scaling: {X_train_scaled.std(axis=0)}")
    
    # Optionally run BIC optimization
    if run_bic_optimization:
        logger.info("\nRunning BIC optimization to validate 3-state choice...")
        
        bic_result = select_optimal_states(
            X_train_scaled,
            min_states=2,
            max_states=5,
            covariance_type='full',
            n_iter=1000,
            random_state=42
        )
        
        logger.info(f"Optimal states from BIC: {bic_result['optimal_n']}")
        logger.info(f"BIC scores: {bic_result['bic_scores']}")
        
        # Save BIC curve
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        bic_fig_path = os.path.join(config.FIGURES_DIR, 'bic_selection.png')
        plot_bic_curve(bic_result['bic_scores'], save_path=bic_fig_path)
        
        if bic_result['optimal_n'] != 3:
            logger.warning(
                f"BIC suggests {bic_result['optimal_n']} states, but using 3 for financial interpretation"
            )
    
    # Initialize and train RegimeDetector
    logger.info("\nTraining RegimeDetector with 3 components...")
    
    detector = RegimeDetector(
        n_components=3,
        covariance_type='full',
        n_iter=1000,
        random_state=42
    )
    
    # Store scaler in detector for later use
    detector.scaler = scaler
    detector.feature_columns = hmm_feature_cols
    
    detector.fit(X_train_scaled)
    
    # Predict on training data for validation
    train_regimes = detector.predict(X_train_scaled)
    
    # Validate training results
    validation_passed = validate_training_data(X_train_scaled, train_regimes)
    
    # Validate transition matrix
    transmat = detector.get_transition_matrix()
    row_sums = transmat.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        logger.error(f"Transition matrix rows do not sum to 1.0: {row_sums}")
    else:
        logger.info("✓ Transition matrix validation passed")
    
    # Log transition probabilities
    logger.info("\nTransition Matrix:")
    transmat_df = detector.calculate_transition_probabilities()
    logger.info(f"\n{transmat_df}")
    
    # Log stationary distribution
    logger.info("\nStationary Distribution:")
    stationary = detector.get_stationary_distribution()
    for regime, prob in stationary.items():
        logger.info(f"  {regime}: {prob:.1%}")
    
    # Predict on full dataset for downstream phases
    logger.info("\nPredicting regimes on full dataset...")
    
    # Get full feature set with valid dates
    X_full_raw = features[hmm_feature_cols].dropna()
    # CRITICAL: Scale full data with same scaler
    X_full_scaled = scaler.transform(X_full_raw.values)
    full_regimes = detector.predict(X_full_scaled)
    full_confidence = detector.get_regime_confidence(X_full_scaled)
    
    # Calculate regime summary statistics
    logger.info("\nRegime Summary Statistics (Training Period):")
    summary_stats = get_regime_summary_statistics(
        train_features.loc[X_train_raw.index],
        train_regimes,
        returns_column='log_returns'
    )
    logger.info(f"\n{summary_stats.to_string()}")
    
    # Check average confidence
    avg_confidence = full_confidence.mean()
    logger.info(f"\nAverage regime confidence: {avg_confidence:.2%}")
    if avg_confidence < 0.7:
        logger.warning(f"Average confidence ({avg_confidence:.2%}) is below 0.7 threshold")
    
    # Save model
    if save_model:
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        model_path = os.path.join(config.MODELS_DIR, 'regime_detector.pkl')
        
        metadata = {
            'train_start': config.TRAIN_START,
            'train_end': config.TRAIN_END,
            'n_train_samples': len(X_train_raw),
            'feature_columns': hmm_feature_cols,
            'avg_confidence': float(avg_confidence),
            'validation_passed': validation_passed
        }
        
        detector.save_model(model_path, metadata=metadata)
        logger.info(f"Model saved to {model_path}")
    
    # Save regime labels for downstream phases
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    regime_labels_path = os.path.join(config.PROCESSED_DATA_DIR, 'regime_labels.csv')
    
    regime_df = pd.DataFrame({
        'date': X_full_raw.index,
        'regime_id': full_regimes,
        'regime_name': label_regimes(full_regimes),
        'confidence': full_confidence
    })
    regime_df.to_csv(regime_labels_path, index=False)
    logger.info(f"Regime labels saved to {regime_labels_path}")
    
    # Export comprehensive regime analysis
    regime_analysis_dir = os.path.join(config.OUTPUTS_DIR, 'regime_analysis')
    exported_files = export_regime_analysis(
        detector,
        X_full_scaled,
        X_full_raw.index,
        output_dir=regime_analysis_dir
    )
    
    logger.info("="*60)
    logger.info("HMM REGIME DETECTOR TRAINING COMPLETE")
    logger.info("="*60)
    
    return {
        'detector': detector,
        'train_regimes': train_regimes,
        'full_regimes': full_regimes,
        'regime_confidence': full_confidence,
        'summary_statistics': summary_stats,
        'exported_files': exported_files,
        'validation_passed': validation_passed,
        'model_path': model_path if save_model else None,
        'regime_labels_path': regime_labels_path
    }


def main():
    """Main entry point for training script."""
    try:
        result = train_regime_detector(
            run_bic_optimization=True,
            save_model=True
        )
        
        if result['validation_passed']:
            logger.info("\n✓ Training completed successfully!")
        else:
            logger.warning("\n⚠ Training completed with validation warnings")
        
        return result
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
