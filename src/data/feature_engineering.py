"""
Feature Engineering Module

Implements comprehensive feature engineering with strict anti-leakage enforcement.
ALL features are shifted by 1 period to prevent lookahead bias.
"""

import logging
from typing import Dict, List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering with mandatory shift enforcement.
    
    All feature methods MUST return shifted data to prevent lookahead bias.
    The .shift(1) ensures that features computed at market close on day t
    are used to predict day t+1.
    """
    
    def __init__(self, config):
        """
        Initialize FeatureEngineer with configuration.
        
        Args:
            config: Configuration module or object
        """
        self.config = config
        self.features_computed = []
    
    def compute_log_returns(self, close_prices: pd.Series) -> pd.Series:
        """
        Compute log returns (shifted).
        
        Args:
            close_prices: Series of closing prices
            
        Returns:
            Shifted log returns
        """
        log_returns = np.log(close_prices / close_prices.shift(1))
        
        # CRITICAL: Shift by 1 to prevent lookahead bias
        # Returns computed at close of day t are used for prediction on day t+1
        return log_returns.shift(1)
    
    def compute_parkinson_volatility(
        self,
        high: pd.Series,
        low: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Compute Parkinson volatility estimator (shifted).
        
        More efficient than close-to-close volatility.
        
        Args:
            high: High prices
            low: Low prices
            window: Rolling window size
            
        Returns:
            Shifted Parkinson volatility
        """
        # Parkinson volatility formula
        hl_ratio = np.log(high / low)
        parkinson_vol = np.sqrt((hl_ratio ** 2) / (4 * np.log(2)))
        
        # Apply rolling mean
        rolling_vol = parkinson_vol.rolling(
            window=window,
            min_periods=window  # Enforce explicit min_periods
        ).mean()
        
        # CRITICAL: Shift by 1
        return rolling_vol.shift(1)
    
    def compute_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Compute Relative Strength Index (shifted).
        
        Args:
            close: Closing prices
            period: RSI period
            
        Returns:
            Shifted RSI values (0-100)
        """
        # Calculate price changes
        delta = close.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period, min_periods=period).mean()
        avg_losses = losses.rolling(window=period, min_periods=period).mean()
        
        # Calculate RS and RSI with proper edge case handling
        # When avg_losses = 0, RSI = 100 (all gains, no losses)
        rs = avg_gains / avg_losses
        rsi = np.where(avg_losses == 0, 100, 100 - (100 / (1 + rs)))
        rsi = pd.Series(rsi, index=close.index)
        
        # Handle remaining NaN values (e.g., insufficient data)
        rsi = rsi.fillna(50)  # Neutral RSI for missing data
        
        # CRITICAL: Shift by 1
        return rsi.shift(1)
    
    def compute_macd(
        self,
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute MACD indicator (shifted).
        
        Args:
            close: Closing prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (macd_line, signal_line, histogram) - all shifted
        """
        # Calculate EMAs
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        # CRITICAL: Shift all components by 1
        return (
            macd_line.shift(1),
            signal_line.shift(1),
            histogram.shift(1)
        )
    
    def compute_bollinger_bands(
        self,
        close: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> pd.Series:
        """
        Compute Bollinger Band width (shifted).
        
        Returns normalized width: (upper - lower) / middle
        
        Args:
            close: Closing prices
            window: Window for moving average
            num_std: Number of standard deviations
            
        Returns:
            Shifted Bollinger Band width
        """
        # Middle band (SMA)
        middle_band = close.rolling(window=window, min_periods=window).mean()
        
        # Standard deviation
        std = close.rolling(window=window, min_periods=window).std()
        
        # Upper and lower bands
        upper_band = middle_band + (num_std * std)
        lower_band = middle_band - (num_std * std)
        
        # Normalized width
        bb_width = (upper_band - lower_band) / middle_band
        
        # CRITICAL: Shift by 1
        return bb_width.shift(1)
    
    def compute_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Compute Average True Range (shifted).
        
        Args:
            high: High prices
            low: Low prices
            close: Closing prices
            period: ATR period
            
        Returns:
            Shifted ATR values
        """
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is rolling mean of True Range
        atr = true_range.rolling(window=period, min_periods=period).mean()
        
        # CRITICAL: Shift by 1
        return atr.shift(1)
    
    def compute_rolling_sharpe(
        self,
        returns: pd.Series,
        window: int = 63,
        risk_free_rate: float = 0.02
    ) -> pd.Series:
        """
        Compute rolling Sharpe ratio (shifted).
        
        Args:
            returns: Log returns
            window: Rolling window (default 63 = 3 months)
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Shifted annualized Sharpe ratio
        """
        # Daily risk-free rate
        daily_rf = risk_free_rate / 252
        
        # Excess returns
        excess_returns = returns - daily_rf
        
        # Rolling mean and std
        rolling_mean = excess_returns.rolling(
            window=window,
            min_periods=window
        ).mean()
        
        rolling_std = excess_returns.rolling(
            window=window,
            min_periods=window
        ).std()
        
        # Annualized Sharpe ratio
        sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        
        # CRITICAL: Shift by 1
        return sharpe.shift(1)
    
    def compute_rolling_max_drawdown(
        self,
        close: pd.Series,
        window: int = 126
    ) -> pd.Series:
        """
        Compute rolling maximum drawdown (shifted).
        
        Args:
            close: Closing prices
            window: Rolling window (default 126 = 6 months)
            
        Returns:
            Shifted maximum drawdown (negative values)
        """
        # Rolling maximum
        rolling_max = close.rolling(window=window, min_periods=window).max()
        
        # Drawdown
        drawdown = (close - rolling_max) / rolling_max
        
        # Maximum drawdown in window
        max_dd = drawdown.rolling(window=window, min_periods=window).min()
        
        # CRITICAL: Shift by 1
        return max_dd.shift(1)
    
    def compute_vix_percentile(
        self,
        vix: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Compute VIX percentile rank (shifted).
        
        Args:
            vix: VIX values
            window: Rolling window for percentile calculation
            
        Returns:
            Shifted percentile rank (0-1)
        """
        # Calculate rolling percentile rank
        percentile = vix.rolling(window=window, min_periods=window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False
        )
        
        # CRITICAL: Shift by 1
        return percentile.shift(1)
    
    def compute_yield_curve_slope(
        self,
        ten_year: pd.Series,
        two_year: pd.Series
    ) -> pd.Series:
        """
        Compute yield curve slope (shifted).
        
        Args:
            ten_year: 10-year treasury yield
            two_year: 2-year treasury yield
            
        Returns:
            Shifted yield curve slope
        """
        slope = ten_year - two_year
        
        # CRITICAL: Shift by 1
        return slope.shift(1)
    
    def compute_moving_averages(
        self,
        close: pd.Series,
        sma_windows: List[int],
        ema_windows: List[int]
    ) -> Dict[str, pd.Series]:
        """
        Compute multiple moving averages (shifted).
        
        Args:
            close: Closing prices
            sma_windows: List of SMA window sizes
            ema_windows: List of EMA window sizes
            
        Returns:
            Dictionary of shifted moving averages
        """
        mas = {}
        
        # Simple Moving Averages
        for window in sma_windows:
            sma = close.rolling(window=window, min_periods=window).mean()
            mas[f'SMA_{window}'] = sma.shift(1)  # CRITICAL: Shift by 1
        
        # Exponential Moving Averages
        for window in ema_windows:
            ema = close.ewm(span=window, adjust=False).mean()
            mas[f'EMA_{window}'] = ema.shift(1)  # CRITICAL: Shift by 1
        
        return mas
    
    def compute_price_momentum(
        self,
        close: pd.Series,
        periods: List[int] = [5, 10, 20]
    ) -> Dict[str, pd.Series]:
        """
        Compute price momentum over multiple periods (shifted).
        
        Args:
            close: Closing prices
            periods: List of lookback periods
            
        Returns:
            Dictionary of shifted momentum features
        """
        momentum = {}
        
        for period in periods:
            mom = (close / close.shift(period)) - 1
            momentum[f'momentum_{period}d'] = mom.shift(1)  # CRITICAL: Shift by 1
        
        return momentum
    
    def _validate_no_leakage(
        self,
        features_df: pd.DataFrame,
        original_df: pd.DataFrame
    ) -> bool:
        """
        Validate that no feature contains lookahead information.
        
        Args:
            features_df: Engineered features
            original_df: Original OHLCV data
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If leakage is detected
        """
        logger.info("Running anti-leakage validation...")
        
        # Check that features are properly aligned
        if not features_df.index.equals(original_df.index):
            # Features should have same or fewer rows (due to NaN dropping)
            if not set(features_df.index).issubset(set(original_df.index)):
                raise ValueError("Feature index contains dates not in original data")
        
        # For each feature, verify first valid value comes AFTER original data
        for col in features_df.columns:
            feature_first_valid = features_df[col].first_valid_index()
            
            if feature_first_valid is not None:
                # Get original data at that date
                if feature_first_valid in original_df.index:
                    original_idx = original_df.index.get_loc(feature_first_valid)
                    
                    # Feature at time t should use data from at least t-1
                    if original_idx == 0:
                        # First date in original - feature should be NaN
                        if not pd.isna(features_df[col].iloc[0]):
                            raise ValueError(
                                f"Feature {col} has value at first date - "
                                f"indicates no shift was applied"
                            )
        
        logger.info("✓ Anti-leakage validation passed")
        return True
    
    def engineer_features(
        self,
        data_df: pd.DataFrame,
        primary_ticker: str = 'SPY'
    ) -> pd.DataFrame:
        """
        Engineer all features from raw OHLCV data.
        
        Args:
            data_df: DataFrame with MultiIndex columns (Ticker, OHLCV)
            primary_ticker: Primary ticker for prediction target
            
        Returns:
            DataFrame with all engineered features (shifted)
        """
        logger.info("Starting feature engineering...")
        
        features = {}
        
        # Extract primary ticker data
        if primary_ticker not in data_df.columns.get_level_values(0):
            raise ValueError(f"Primary ticker {primary_ticker} not found in data")
        
        close = data_df[primary_ticker]['Close']
        high = data_df[primary_ticker]['High']
        low = data_df[primary_ticker]['Low']
        volume = data_df[primary_ticker]['Volume']
        
        # Log returns
        logger.info("Computing log returns...")
        features['log_returns'] = self.compute_log_returns(close)
        
        # Parkinson volatility
        logger.info("Computing Parkinson volatility...")
        features['parkinson_vol'] = self.compute_parkinson_volatility(
            high, low, window=self.config.PARKINSON_VOL_WINDOW
        )
        
        # RSI
        logger.info("Computing RSI...")
        features['rsi'] = self.compute_rsi(close, period=self.config.RSI_PERIOD)
        
        # MACD
        logger.info("Computing MACD...")
        macd, signal, hist = self.compute_macd(
            close,
            fast=self.config.MACD_FAST,
            slow=self.config.MACD_SLOW,
            signal=self.config.MACD_SIGNAL
        )
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = hist
        
        # Bollinger Bands
        logger.info("Computing Bollinger Bands...")
        features['bb_width'] = self.compute_bollinger_bands(
            close,
            window=self.config.BOLLINGER_BAND_WINDOW,
            num_std=self.config.BOLLINGER_BAND_STD
        )
        
        # ATR
        logger.info("Computing ATR...")
        features['atr'] = self.compute_atr(
            high, low, close, period=self.config.ATR_PERIOD
        )
        
        # Rolling Sharpe
        logger.info("Computing rolling Sharpe ratio...")
        returns = np.log(close / close.shift(1))
        features['rolling_sharpe'] = self.compute_rolling_sharpe(
            returns,
            window=self.config.ROLLING_SHARPE_WINDOW,
            risk_free_rate=self.config.RISK_FREE_RATE
        )
        
        # Rolling max drawdown
        logger.info("Computing rolling maximum drawdown...")
        features['rolling_max_dd'] = self.compute_rolling_max_drawdown(
            close, window=self.config.ROLLING_MAX_DD_WINDOW
        )
        
        # Moving averages
        logger.info("Computing moving averages...")
        mas = self.compute_moving_averages(
            close,
            sma_windows=self.config.SMA_WINDOWS,
            ema_windows=self.config.EMA_WINDOWS
        )
        features.update(mas)
        
        # Price momentum
        logger.info("Computing price momentum...")
        momentum = self.compute_price_momentum(close)
        features.update(momentum)
        
        # VIX features if available
        if '^VIX' in data_df.columns.get_level_values(0):
            logger.info("Computing VIX features...")
            vix = data_df['^VIX']['Close']
            features['vix_level'] = vix.shift(1)  # CRITICAL: Shift by 1
            features['vix_percentile'] = self.compute_vix_percentile(
                vix, window=self.config.VIX_PERCENTILE_WINDOW
            )
        
        # Volume features
        logger.info("Computing volume features...")
        volume_ma = volume.rolling(window=20, min_periods=20).mean()
        features['volume_ratio'] = (volume / volume_ma).shift(1)  # CRITICAL: Shift by 1
        
        # Create features DataFrame
        features_df = pd.DataFrame(features, index=data_df.index)
        
        # Validate no leakage
        if self.config.ENFORCE_SHIFT:
            self._validate_no_leakage(features_df, data_df)
        
        # Drop rows with NaN values (from initial rolling windows)
        initial_rows = len(features_df)
        features_df = features_df.dropna()
        dropped_rows = initial_rows - len(features_df)
        
        logger.info(f"Dropped {dropped_rows} rows with NaN values")
        logger.info(f"Final feature matrix shape: {features_df.shape}")
        logger.info(f"Features: {list(features_df.columns)}")
        
        return features_df


# Example usage
if __name__ == '__main__':
    from config import (
        TICKERS, DATA_START_DATE, DATA_END_DATE,
        FEATURES_FILE, RAW_DATA_DIR
    )
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader(raw_data_dir=RAW_DATA_DIR)
    data = loader.get_data(list(TICKERS.keys()), DATA_START_DATE, DATA_END_DATE)
    
    # Engineer features
    import config
    engineer = FeatureEngineer(config)
    features = engineer.engineer_features(data)
    
    # Save features
    import os
    os.makedirs(os.path.dirname(FEATURES_FILE), exist_ok=True)
    features.to_pickle(FEATURES_FILE)
    logger.info(f"Features saved to {FEATURES_FILE}")
