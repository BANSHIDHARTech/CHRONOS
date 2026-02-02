"""
Data Pipeline Orchestration

Main entry point for the CHRONOS data pipeline.
Coordinates data loading, feature engineering, target creation, and splitting.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import config
from src.data.data_loader import DataLoader
from src.data.feature_engineering import FeatureEngineer
from src.data.target_engineering import TargetEngineer
from src.data.data_splitter import DataSplitter
from src.utils.validation import validate_pipeline_integrity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_data_pipeline(
    tickers: Optional[list] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
    validate: bool = True,
    save_intermediate: bool = True,
    verbose: bool = False
) -> Dict:
    """
    Run the complete data pipeline.
    
    Args:
        tickers: List of ticker symbols (uses config if None)
        start_date: Start date YYYY-MM-DD (uses config if None)
        end_date: End date YYYY-MM-DD (uses config if None)
        use_cache: Whether to use cached data
        validate: Whether to run comprehensive validation
        save_intermediate: Whether to save intermediate results
        verbose: Whether to enable verbose logging
        
    Returns:
        Dictionary containing:
        - raw_data: Raw OHLCV data
        - features: Engineered features
        - targets: Target variables
        - splits: Train/val/test splits
        - metadata: Pipeline metadata
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("CHRONOS DATA PIPELINE")
    logger.info("=" * 80)
    
    pipeline_start_time = datetime.now()
    
    # Use config defaults if not specified
    if tickers is None:
        tickers = list(config.TICKERS.keys())
    if start_date is None:
        start_date = config.DATA_START_DATE
    if end_date is None:
        end_date = config.DATA_END_DATE
    
    logger.info(f"\nPipeline Configuration:")
    logger.info(f"  Tickers: {', '.join(tickers)}")
    logger.info(f"  Date Range: {start_date} to {end_date}")
    logger.info(f"  Use Cache: {use_cache}")
    logger.info(f"  Validate: {validate}")
    
    # Step 1: Load raw data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOADING RAW DATA")
    logger.info("=" * 80)
    
    loader = DataLoader(raw_data_dir=config.RAW_DATA_DIR)
    raw_data = loader.get_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache
    )
    
    logger.info(f"✓ Loaded raw data: {raw_data.shape}")
    
    # Step 2: Engineer features
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: ENGINEERING FEATURES")
    logger.info("=" * 80)
    
    feature_engineer = FeatureEngineer(config)
    features = feature_engineer.engineer_features(raw_data, primary_ticker='SPY')
    
    logger.info(f"✓ Engineered {features.shape[1]} features from {features.shape[0]} samples")
    
    if save_intermediate:
        features.to_pickle(config.FEATURES_FILE)
        logger.info(f"✓ Saved features to {config.FEATURES_FILE}")
    
    # Step 3: Engineer targets
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: ENGINEERING TARGETS")
    logger.info("=" * 80)
    
    target_engineer = TargetEngineer(config)
    
    # Extract primary ticker close prices
    primary_ticker = 'SPY'
    if primary_ticker not in raw_data.columns.get_level_values(0):
        logger.warning(f"{primary_ticker} not found, using first available ticker")
        primary_ticker = raw_data.columns.get_level_values(0)[0]
    
    close_prices = raw_data[primary_ticker]['Close']
    targets = target_engineer.engineer_targets(close_prices)
    
    logger.info(f"✓ Engineered targets: {targets.shape}")
    
    if save_intermediate:
        targets.to_pickle(config.TARGETS_FILE)
        logger.info(f"✓ Saved targets to {config.TARGETS_FILE}")
    
    # Step 4: Align features and targets
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: ALIGNING FEATURES AND TARGETS")
    logger.info("=" * 80)
    
    X, y = target_engineer.align_features_and_targets(features, targets)
    
    logger.info(f"✓ Aligned data: X={X.shape}, y={y.shape}")
    
    if save_intermediate:
        aligned_data = pd.concat([X, y.rename('quintile_label')], axis=1)
        aligned_data.to_pickle(config.ALIGNED_DATA_FILE)
        logger.info(f"✓ Saved aligned data to {config.ALIGNED_DATA_FILE}")
    
    # Step 5: Split into train/val/test
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: SPLITTING DATA")
    logger.info("=" * 80)
    
    splitter = DataSplitter(config)
    splits = splitter.split_with_config(X, y)
    
    logger.info(f"✓ Created train/val/test splits")
    
    if save_intermediate:
        splitter.save_splits(splits, config.PROCESSED_DATA_DIR)
        logger.info(f"✓ Saved splits to {config.PROCESSED_DATA_DIR}")
    
    # Step 6: Validate pipeline
    if validate:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: VALIDATING PIPELINE INTEGRITY")
        logger.info("=" * 80)
        
        validation_passed = validate_pipeline_integrity(
            raw_data=raw_data,
            features=features,
            targets=targets,
            splits=splits
        )
        
        if not validation_passed:
            logger.error("✗ Pipeline validation FAILED")
            logger.error("Review the validation errors above before proceeding")
        else:
            logger.info("✓ Pipeline validation PASSED")
    
    # Calculate pipeline execution time
    pipeline_end_time = datetime.now()
    execution_time = (pipeline_end_time - pipeline_start_time).total_seconds()
    
    # Prepare metadata
    metadata = {
        'tickers': tickers,
        'start_date': start_date,
        'end_date': end_date,
        'num_features': features.shape[1],
        'num_samples': {
            'train': len(splits['train'][0]),
            'val': len(splits['val'][0]),
            'test': len(splits['test'][0])
        },
        'execution_time_seconds': execution_time,
        'timestamp': pipeline_end_time.isoformat(),
        'config': {
            'forward_return_days': config.FORWARD_RETURN_DAYS,
            'rsi_period': config.RSI_PERIOD,
            'enforce_shift': config.ENFORCE_SHIFT
        }
    }
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Execution Time: {execution_time:.2f} seconds")
    logger.info(f"Raw Data Shape: {raw_data.shape}")
    logger.info(f"Features Shape: {features.shape}")
    logger.info(f"Targets Shape: {targets.shape}")
    logger.info(f"Train Samples: {metadata['num_samples']['train']}")
    logger.info(f"Val Samples: {metadata['num_samples']['val']}")
    logger.info(f"Test Samples: {metadata['num_samples']['test']}")
    logger.info("=" * 80)
    
    return {
        'raw_data': raw_data,
        'features': features,
        'targets': targets,
        'splits': splits,
        'metadata': metadata
    }


def main():
    """
    Command-line interface for the data pipeline.
    """
    parser = argparse.ArgumentParser(
        description='CHRONOS Data Pipeline - Institutional-grade quantitative finance data processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python -m src.data.pipeline
  
  # Run with custom date range
  python -m src.data.pipeline --start 2020-01-01 --end 2024-12-31
  
  # Run without cache
  python -m src.data.pipeline --no-cache
  
  # Run with verbose logging
  python -m src.data.pipeline --verbose
  
  # Run without validation (faster)
  python -m src.data.pipeline --no-validate
        """
    )
    
    parser.add_argument(
        '--start',
        type=str,
        default=None,
        help=f'Start date (YYYY-MM-DD), default: {config.DATA_START_DATE}'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help=f'End date (YYYY-MM-DD), default: {config.DATA_END_DATE}'
    )
    
    parser.add_argument(
        '--tickers',
        type=str,
        nargs='+',
        default=None,
        help=f'List of ticker symbols, default: {list(config.TICKERS.keys())}'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Download fresh data instead of using cache'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip comprehensive validation (faster but less safe)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save intermediate results'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate configuration
    try:
        config.validate_all()
        logger.info("✓ Configuration validation passed")
    except ValueError as e:
        logger.error(f"✗ Configuration validation failed: {e}")
        sys.exit(1)
    
    # Run pipeline
    try:
        results = run_data_pipeline(
            tickers=args.tickers,
            start_date=args.start,
            end_date=args.end,
            use_cache=not args.no_cache,
            validate=not args.no_validate,
            save_intermediate=not args.no_save,
            verbose=args.verbose
        )
        
        logger.info("\n✓ Pipeline completed successfully!")
        
        # Print next steps
        logger.info("\nNext Steps:")
        logger.info("1. Review the validation results above")
        logger.info("2. Explore the processed data in data/processed/")
        logger.info("3. Proceed to model training (Phase 2)")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed with error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == '__main__':
    main()
