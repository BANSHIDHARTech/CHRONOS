"""
HMM Regime Detector Module

Implements Gaussian Hidden Markov Model for market regime detection.
Identifies three distinct market states: Euphoria (bull), Complacency (neutral), 
and Capitulation (bear).
"""

import logging
import pickle
import json
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy import linalg

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Gaussian Hidden Markov Model for market regime detection.
    
    Attributes:
        n_components: Number of hidden states (regimes)
        covariance_type: Type of covariance parameters ('full', 'diag', 'spherical', 'tied')
        n_iter: Maximum number of EM iterations
        random_state: Random seed for reproducibility
        model: Fitted GaussianHMM instance
    """
    
    # Regime name mapping
    REGIME_NAMES = {
        0: 'Euphoria',
        1: 'Complacency',
        2: 'Capitulation'
    }
    
    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = 'full',
        n_iter: int = 1000,
        random_state: int = 42,
        tol: float = 1e-4
    ):
        """
        Initialize RegimeDetector.
        
        Args:
            n_components: Number of hidden states (Euphoria, Complacency, Capitulation)
            covariance_type: Covariance type for regime-specific volatility patterns
            n_iter: Maximum EM algorithm iterations
            random_state: Random seed for reproducibility
            tol: Convergence tolerance
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol
        
        self.model: Optional[GaussianHMM] = None
        self.is_fitted = False
        self.training_info: Dict[str, Any] = {}
        self.feature_names: Optional[list] = None
        
        logger.info(
            f"Initialized RegimeDetector with {n_components} components, "
            f"{covariance_type} covariance"
        )
    
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """
        Validate input data for HMM.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Validated numpy array
            
        Raises:
            ValueError: If input is invalid
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        elif isinstance(X, pd.Series):
            X = X.values.reshape(-1, 1)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")
        
        # Check for NaN values
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            raise ValueError(f"Input contains {nan_count} NaN values. Remove before fitting.")
        
        # Check for infinite values
        if np.isinf(X).any():
            inf_count = np.isinf(X).sum()
            raise ValueError(f"Input contains {inf_count} infinite values.")
        
        return X
    
    def fit(self, X_train: np.ndarray) -> 'RegimeDetector':
        """
        Fit the HMM on training data.
        
        Args:
            X_train: Training features (n_samples, n_features)
                     Typically log_returns and parkinson_vol
        
        Returns:
            self: Fitted RegimeDetector instance
        """
        logger.info("Fitting HMM Regime Detector...")
        
        X = self._validate_input(X_train)
        
        logger.info(f"Training data shape: {X.shape}")
        
        # Initialize HMM
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            tol=self.tol,
            verbose=False
        )
        
        # Fit model
        self.model.fit(X)
        
        # CRITICAL: Reorder regimes based on financial characteristics
        # This ensures: 0=Euphoria (high returns), 1=Complacency (mid), 2=Capitulation (low returns)
        self._reorder_regimes_by_returns()
        
        # Store training statistics
        self.training_info = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'converged': self.model.monitor_.converged,
            'n_iter_used': self.model.monitor_.iter,
            'final_log_likelihood': self.model.score(X),
            'feature_names': self.feature_names
        }
        
        self.is_fitted = True
        
        # Log training results
        logger.info(f"HMM fitting completed:")
        logger.info(f"  Converged: {self.training_info['converged']}")
        logger.info(f"  Iterations: {self.training_info['n_iter_used']}")
        logger.info(f"  Log-likelihood: {self.training_info['final_log_likelihood']:.4f}")
        
        # Validate transition matrix
        self._validate_transition_matrix()
        
        return self
    
    def _reorder_regimes_by_returns(self) -> None:
        """
        Reorder regime labels based on risk-adjusted financial signature.
        
        HMM assigns arbitrary labels (0,1,2). This method reorders robustly by 
        using Sharpe-like ratio (mean/volatility) which is more stable than returns alone.
        
        - Regime 0 = Euphoria (highest risk-adjusted returns)
        - Regime 1 = Complacency (middle risk-adjusted returns)
        - Regime 2 = Capitulation (lowest/negative risk-adjusted returns)
        """
        means = self.model.means_
        covars = self.model.covars_
        
        # Calculate risk-adjusted signature: mean / sqrt(variance)
        # This is more robust than returns alone for distinguishing regimes
        signatures = []
        for i in range(self.n_components):
            mean_ret = means[i, 0]  # log_returns mean (first feature)
            # Handle full covariance matrix
            if len(covars.shape) == 3:
                volatility = np.sqrt(covars[i, 0, 0])  # variance of log_returns
            else:
                volatility = np.sqrt(covars[i, 0])  # diagonal covariance
            
            sharpe_like = mean_ret / (volatility + 1e-8)
            signatures.append({
                'idx': i,
                'mean': mean_ret,
                'vol': volatility,
                'sharpe': sharpe_like
            })
        
        # Sort: Highest Sharpe-like = Euphoria (0), Lowest = Capitulation (2)
        signatures.sort(key=lambda x: x['sharpe'], reverse=True)
        order = np.array([s['idx'] for s in signatures])
        
        logger.info("Reordering regimes by risk-adjusted signature (Sharpe-like):")
        for i, sig in enumerate(signatures):
            regime_name = self.REGIME_NAMES[i]
            logger.info(f"  {regime_name}: mean={sig['mean']:.5f}, vol={sig['vol']:.5f}, sharpe={sig['sharpe']:.3f}")
        
        # Permute model parameters
        self.model.means_ = self.model.means_[order]
        self.model.covars_ = self.model.covars_[order]
        self.model.startprob_ = self.model.startprob_[order]
        
        # Permute transition matrix (both rows and columns)
        self.model.transmat_ = self.model.transmat_[order][:, order]
        
        logger.info("✓ Regime reordering complete")
    
    def _validate_transition_matrix(self) -> bool:
        """
        Validate that transition matrix rows sum to 1.0.
        
        Returns:
            True if valid
            
        Raises:
            ValueError: If transition matrix is invalid
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        transmat = self.model.transmat_
        row_sums = transmat.sum(axis=1)
        
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError(
                f"Transition matrix rows do not sum to 1.0: {row_sums}"
            )
        
        logger.info("✓ Transition matrix validation passed")
        return True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict most likely regime sequence using Viterbi algorithm.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of regime labels (0, 1, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._validate_input(X)
        
        # Use Viterbi algorithm for most likely sequence
        regimes = self.model.predict(X)
        
        return regimes
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate posterior probabilities for all regimes.
        
        Uses forward-backward algorithm to compute P(state | observations).
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Probability matrix (n_samples, n_regimes)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._validate_input(X)
        
        # Get posterior probabilities
        posteriors = self.model.predict_proba(X)
        
        return posteriors
    
    def get_regime_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Extract maximum posterior probability per observation.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of confidence scores (0-1 scale)
        """
        posteriors = self.predict_proba(X)
        confidence = posteriors.max(axis=1)
        return confidence
    
    def get_regime_confidence_series(self, X: np.ndarray, index: Optional[pd.DatetimeIndex] = None) -> pd.Series:
        """
        Calculate posterior probabilities and return as Series.
        
        Args:
            X: Feature matrix
            index: Optional DatetimeIndex for the Series
            
        Returns:
            Series with confidence scores
        """
        confidence = self.get_regime_confidence(X)
        
        if index is not None:
            return pd.Series(confidence, index=index, name='regime_confidence')
        return pd.Series(confidence, name='regime_confidence')
    
    def identify_low_confidence_periods(
        self,
        X: np.ndarray,
        threshold: float = 0.6
    ) -> np.ndarray:
        """
        Flag observations with low regime confidence.
        
        These are regime transition periods with high uncertainty.
        
        Args:
            X: Feature matrix
            threshold: Confidence threshold (default 0.6)
            
        Returns:
            Boolean mask where True = low confidence
        """
        confidence = self.get_regime_confidence(X)
        low_confidence = confidence < threshold
        
        low_conf_pct = low_confidence.mean() * 100
        logger.info(f"Low confidence periods (<{threshold}): {low_conf_pct:.1f}%")
        
        return low_confidence
    
    def get_regime_entropy(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate Shannon entropy of posterior probabilities.
        
        High entropy = uncertain regime classification.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of entropy values
        """
        posteriors = self.predict_proba(X)
        
        # Shannon entropy: -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        entropy = -np.sum(posteriors * np.log(posteriors + eps), axis=1)
        
        return entropy
    
    @property
    def transition_matrix(self) -> np.ndarray:
        """
        Get the transition probability matrix.
        
        Returns:
            3x3 transition probability matrix
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.transmat_
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Return transition matrix (3x3 transition probability matrix).
        
        Returns:
            Transition matrix from self.model.transmat_
        """
        return self.transition_matrix
    
    def calculate_transition_probabilities(self) -> pd.DataFrame:
        """
        Get formatted transition matrix with regime labels.
        
        Returns:
            DataFrame with regime names as index/columns
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        transmat = self.model.transmat_
        regime_names = [self.REGIME_NAMES[i] for i in range(self.n_components)]
        
        df = pd.DataFrame(
            transmat,
            index=regime_names,
            columns=regime_names
        )
        
        return df
    
    def get_stationary_distribution(self) -> Dict[str, float]:
        """
        Calculate long-run regime probabilities.
        
        Finds eigenvector of transition matrix with eigenvalue 1.0,
        representing the stationary distribution.
        
        Returns:
            Dictionary with regime names and long-run probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        transmat = self.model.transmat_
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = linalg.eig(transmat.T)
        
        # Find eigenvector corresponding to eigenvalue 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        
        # Normalize to sum to 1
        stationary = stationary / stationary.sum()
        
        # Create result dictionary
        result = {
            self.REGIME_NAMES[i]: float(stationary[i])
            for i in range(self.n_components)
        }
        
        logger.info("Stationary distribution:")
        for regime, prob in result.items():
            logger.info(f"  {regime}: {prob:.1%}")
        
        return result
    
    def analyze_transition_patterns(self) -> Dict[str, Any]:
        """
        Identify most/least likely transitions and regime shocks.
        
        Returns:
            Dictionary with transition pattern insights
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        transmat = self.model.transmat_
        regime_names = [self.REGIME_NAMES[i] for i in range(self.n_components)]
        
        # Find self-transition probabilities (persistence)
        persistence = {
            regime_names[i]: float(transmat[i, i])
            for i in range(self.n_components)
        }
        
        # Find most likely transitions (excluding self-transitions)
        transitions = []
        for i in range(self.n_components):
            for j in range(self.n_components):
                if i != j:
                    transitions.append({
                        'from': regime_names[i],
                        'to': regime_names[j],
                        'probability': float(transmat[i, j])
                    })
        
        # Sort by probability
        transitions = sorted(transitions, key=lambda x: x['probability'], reverse=True)
        
        # Identify rare transitions (probability < 5%)
        rare_transitions = [t for t in transitions if t['probability'] < 0.05]
        
        result = {
            'persistence': persistence,
            'most_likely_transition': transitions[0] if transitions else None,
            'least_likely_transition': transitions[-1] if transitions else None,
            'rare_transitions': rare_transitions,
            'all_transitions': transitions
        }
        
        return result
    
    def save_model(
        self,
        filepath: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save fitted HMM model to disk.
        
        Args:
            filepath: Path to save model (.pkl)
            metadata: Optional metadata (training dates, BIC score, etc.)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare save data
        save_data = {
            'model': self.model,
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'n_iter': self.n_iter,
            'random_state': self.random_state,
            'tol': self.tol,
            'is_fitted': self.is_fitted,
            'training_info': self.training_info,
            'feature_names': self.feature_names,
            'scaler': getattr(self, 'scaler', None),  # Save scaler for standardization
            'feature_columns': getattr(self, 'feature_columns', None),  # Save feature columns
            'metadata': metadata or {}
        }
        
        # Save pickle file
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Model saved to {filepath}")
        
        # Save companion JSON with metadata
        json_path = filepath.with_suffix('.json')
        json_metadata = {
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'training_info': {
                k: v for k, v in self.training_info.items()
                if isinstance(v, (int, float, str, bool, list, type(None)))
            },
            'metadata': metadata or {}
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_metadata, f, indent=2, default=str)
        
        logger.info(f"Metadata saved to {json_path}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'RegimeDetector':
        """
        Load fitted HMM model from disk.
        
        Args:
            filepath: Path to saved model (.pkl)
            
        Returns:
            Initialized RegimeDetector instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Create new instance
        detector = cls(
            n_components=save_data['n_components'],
            covariance_type=save_data['covariance_type'],
            n_iter=save_data['n_iter'],
            random_state=save_data['random_state'],
            tol=save_data.get('tol', 1e-4)
        )
        
        # Restore state
        detector.model = save_data['model']
        detector.is_fitted = save_data['is_fitted']
        detector.training_info = save_data['training_info']
        detector.feature_names = save_data.get('feature_names')
        detector.scaler = save_data.get('scaler')  # Restore scaler
        detector.feature_columns = save_data.get('feature_columns')  # Restore feature columns
        
        logger.info(f"Model loaded from {filepath}")
        
        return detector
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model parameters, training info, and performance metrics.
        
        Returns:
            Dictionary with comprehensive model information
        """
        info = {
            'parameters': {
                'n_components': self.n_components,
                'covariance_type': self.covariance_type,
                'n_iter': self.n_iter,
                'random_state': self.random_state,
                'tol': self.tol
            },
            'is_fitted': self.is_fitted,
            'training_info': self.training_info.copy() if self.training_info else {},
            'feature_names': self.feature_names
        }
        
        if self.is_fitted:
            info['transition_matrix'] = self.model.transmat_.tolist()
            info['means'] = self.model.means_.tolist()
            info['stationary_distribution'] = self.get_stationary_distribution()
        
        return info


# Example usage
if __name__ == '__main__':
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 500
    
    # Simulate regime-switching data
    X = np.column_stack([
        np.random.randn(n_samples) * 0.02,  # returns
        np.abs(np.random.randn(n_samples)) * 0.01  # volatility
    ])
    
    # Test the detector
    detector = RegimeDetector(n_components=3)
    detector.fit(X)
    
    # Predict regimes
    regimes = detector.predict(X)
    probabilities = detector.predict_proba(X)
    confidence = detector.get_regime_confidence(X)
    
    print(f"\nRegime distribution: {np.bincount(regimes)}")
    print(f"Average confidence: {confidence.mean():.3f}")
    print(f"\nTransition Matrix:")
    print(detector.calculate_transition_probabilities())
    print(f"\nStationary Distribution:")
    print(detector.get_stationary_distribution())
