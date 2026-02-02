"""
Data Loader Module

Handles downloading market data from yfinance with robust error handling,
caching mechanism, and data validation.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    DataLoader class for downloading and managing market data.
    
    Features:
    - Multi-ticker downloads with yfinance
    - Automatic retry logic with exponential backoff
    - Local CSV caching for offline work and reliability
    - Comprehensive data validation
    """
    
    def __init__(self, raw_data_dir: str = 'data/raw'):
        """
        Initialize DataLoader.
        
        Args:
            raw_data_dir: Directory for storing raw CSV files
        """
        self.raw_data_dir = raw_data_dir
        os.makedirs(raw_data_dir, exist_ok=True)
        
    def _get_cache_filename(self, ticker: str, start_date: str, end_date: str) -> str:
        """
        Generate cache filename for a ticker.
        
        Args:
            ticker: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Full path to cache file
        """
        # Clean ticker symbol for filename (replace special chars)
        clean_ticker = ticker.replace('^', '').replace('=', '_')
        filename = f"{clean_ticker}_{start_date}_{end_date}.csv"
        return os.path.join(self.raw_data_dir, filename)
    
    def download_market_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        Download market data from yfinance with retry logic.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_retries: Maximum number of retry attempts
            
        Returns:
            DataFrame with MultiIndex columns (Ticker, OHLCV)
            
        Raises:
            Exception: If critical tickers fail to download
        """
        logger.info(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        all_data = {}
        failed_tickers = []
        
        for ticker in tickers:
            success = False
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Downloading {ticker} (attempt {attempt + 1}/{max_retries})")
                    
                    # Download data for single ticker
                    ticker_data = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        auto_adjust=False  # Keep original columns
                    )
                    
                    if ticker_data.empty:
                        logger.warning(f"No data returned for {ticker}")
                        continue
                    
                    # yfinance returns MultiIndex columns even for single ticker: ('Close', 'SPY')
                    # Flatten to simple column names: 'Close'
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        ticker_data.columns = ticker_data.columns.get_level_values(0)
                    
                    # Check which columns exist
                    available_cols = ticker_data.columns.tolist()
                    logger.debug(f"{ticker} columns: {available_cols}")
                    
                    # Verify we have essential price data
                    if 'Close' not in available_cols and 'Adj Close' not in available_cols:
                        logger.warning(f"No Close price data for {ticker}")
                        continue
                    
                    # Standardize column names if needed
                    if 'Adj Close' in available_cols and 'Close' not in available_cols:
                        ticker_data['Close'] = ticker_data['Adj Close']
                    
                    # Keep only essential columns in standard order
                    essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    available_cols = [col for col in essential_cols if col in ticker_data.columns]
                    ticker_data = ticker_data[available_cols]
                    
                    # Save to cache with simple format
                    cache_file = self._get_cache_filename(ticker, start_date, end_date)
                    ticker_data.to_csv(cache_file, index_label='Date')
                    logger.info(f"Saved {ticker} to cache: {cache_file}")
                    
                    all_data[ticker] = ticker_data
                    success = True
                    break
                    
                except Exception as e:
                    logger.error(f"Error downloading {ticker} (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        # Exponential backoff
                        wait_time = 2 ** attempt
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
            
            if not success:
                logger.error(f"Failed to download {ticker} after {max_retries} attempts")
                failed_tickers.append(ticker)
        
        # Check if critical tickers failed
        critical_tickers = ['^GSPC', 'SPY']
        critical_failures = [t for t in critical_tickers if t in failed_tickers]
        
        if critical_failures:
            raise Exception(f"Critical tickers failed to download: {critical_failures}")
        
        if failed_tickers:
            logger.warning(f"Failed to download {len(failed_tickers)} tickers: {failed_tickers}")
        
        if not all_data:
            raise Exception("No data was successfully downloaded")
        
        # Combine all ticker data into MultiIndex DataFrame
        # Create proper MultiIndex with (Ticker, Column) structure
        combined_df = pd.concat(all_data, axis=1, keys=all_data.keys())
        
        # Sort by level to organize columns
        combined_df = combined_df.sort_index(axis=1, level=0)
        
        # Log column structure for debugging
        logger.debug(f"Column structure sample: {combined_df.columns.tolist()[:5]}")
        
        logger.info(f"Successfully downloaded {len(all_data)} tickers")
        logger.info(f"Data shape: {combined_df.shape}")
        logger.info(f"Date range: {combined_df.index[0]} to {combined_df.index[-1]}")
        
        return combined_df
    
    def load_from_cache(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Load data from local CSV cache.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with MultiIndex columns or None if cache miss
        """
        logger.info("Attempting to load data from cache...")
        
        all_data = {}
        
        for ticker in tickers:
            cache_file = self._get_cache_filename(ticker, start_date, end_date)
            
            if not os.path.exists(cache_file):
                logger.info(f"Cache miss for {ticker}")
                return None
            
            try:
                ticker_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                # Ensure we have the essential columns
                if 'Close' not in ticker_data.columns:
                    logger.error(f"Cache file for {ticker} missing Close column")
                    return None
                all_data[ticker] = ticker_data
                logger.info(f"Loaded {ticker} from cache")
            except Exception as e:
                logger.error(f"Error loading {ticker} from cache: {e}")
                return None
        
        if not all_data:
            return None
        
        # Combine into MultiIndex DataFrame
        combined_df = pd.concat(all_data, axis=1, keys=all_data.keys())
        combined_df = combined_df.sort_index(axis=1, level=0)
        
        logger.info(f"Successfully loaded {len(all_data)} tickers from cache")
        return combined_df
    
    def get_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get market data (from cache or download).
        
        This is the primary interface method.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to try loading from cache first
            
        Returns:
            DataFrame with MultiIndex columns (Ticker, OHLCV)
        """
        data = None
        
        # Try cache first if enabled
        if use_cache:
            data = self.load_from_cache(tickers, start_date, end_date)
        
        # Download if cache miss or cache disabled
        if data is None:
            data = self.download_market_data(tickers, start_date, end_date)
        
        # Validate data
        validation_report = self.validate_data(data)
        
        if not validation_report['is_valid']:
            logger.warning("Data validation found issues:")
            for issue in validation_report['issues']:
                logger.warning(f"  - {issue}")
        
        # Forward-fill missing values only (never backward-fill)
        logger.info("Forward-filling missing values...")
        data = data.ffill()
        
        return data
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate data completeness and quality.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results:
            - is_valid: bool
            - issues: List[str]
            - statistics: Dict
        """
        issues = []
        statistics = {}
        
        # Check if DataFrame is empty
        if df.empty:
            issues.append("DataFrame is empty")
            return {'is_valid': False, 'issues': issues, 'statistics': statistics}
        
        # Check datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append("Index is not DatetimeIndex")
        
        # Check for duplicate indices
        if df.index.duplicated().any():
            num_duplicates = df.index.duplicated().sum()
            issues.append(f"Found {num_duplicates} duplicate date entries")
        
        # Check minimum history
        num_days = len(df)
        statistics['num_days'] = num_days
        
        if num_days < 252:
            issues.append(f"Insufficient history: {num_days} days (minimum 252 recommended)")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            statistics['missing_values'] = missing_counts[missing_counts > 0].to_dict()
            
            # Calculate percentage
            missing_pct = (missing_counts / len(df) * 100)
            high_missing = missing_pct[missing_pct > 5]
            
            if not high_missing.empty:
                issues.append(f"High missing value percentage (>5%) in columns: {high_missing.to_dict()}")
        
        # Check for gaps in time series (only trading days expected)
        date_diffs = df.index.to_series().diff()
        statistics['max_gap_days'] = date_diffs.max().days if len(date_diffs) > 0 else 0
        
        large_gaps = date_diffs[date_diffs > pd.Timedelta(days=7)]
        if not large_gaps.empty:
            issues.append(f"Found {len(large_gaps)} gaps larger than 7 days in time series")
        
        # Verify OHLCV columns exist for each ticker
        if isinstance(df.columns, pd.MultiIndex):
            tickers = df.columns.get_level_values(0).unique()
            statistics['num_tickers'] = len(tickers)
            
            # Check for essential columns (at minimum need Close price)
            for ticker in tickers:
                ticker_cols = df[ticker].columns.tolist()
                if 'Close' not in ticker_cols:
                    issues.append(f"Ticker {ticker} missing Close price")
        else:
            # Single level columns - check if we have Close at minimum
            if 'Close' not in df.columns:
                issues.append("Missing Close price column")
        
        # Check for negative prices or volumes (handle MultiIndex)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                if isinstance(col, tuple):
                    col_data = df[col]
                else:
                    col_data = df[col]
                if col_data.lt(0).any():
                    issues.append(f"Found negative values in column: {col}")
            except Exception as e:
                logger.debug(f"Skipping negative check for column {col}: {e}")
        
        is_valid = len(issues) == 0
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'statistics': statistics
        }


# Example usage
if __name__ == '__main__':
    from config import TICKERS, DATA_START_DATE, DATA_END_DATE
    
    loader = DataLoader()
    
    # Get data for configured tickers
    ticker_list = list(TICKERS.keys())
    data = loader.get_data(ticker_list, DATA_START_DATE, DATA_END_DATE)
    
    print(f"\nLoaded data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"\nFirst few rows:")
    print(data.head())
