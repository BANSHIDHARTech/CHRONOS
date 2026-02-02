"""
Adaptive Predictor with Regime Routing

This module implements the AdaptivePredictor class that integrates regime
detection with regime-specialized XGBoost models for dynamic prediction
routing based on current market conditions.
"""

import logging
from typing import Dict, Optional, Union, List
import numpy as np

from .regime_detector import RegimeDetector
from .xgboost_ensemble import RegimeSpecializedEnsemble

# Configure logging
logger = logging.getLogger(__name__)


class AdaptivePredictor:
    """
    Dynamic predictor that routes predictions to regime-specialized models.
    
    This class integrates the RegimeDetector (HMM) with the RegimeSpecializedEnsemble
    (XGBoost) to automatically detect the current market regime and route
    predictions to the appropriate specialized model.
    
    Attributes:
        regime_detector (RegimeDetector): Trained HMM for regime detection
        ensemble (RegimeSpecializedEnsemble): Trained regime-specialized models
        fallback_regime (int): Default regime to use if detection fails
    """
    
    def __init__(
        self,
        regime_detector: RegimeDetector,
        ensemble: RegimeSpecializedEnsemble,
        fallback_regime: int = 1
    ):
        """
        Initialize the AdaptivePredictor.
        
        Args:
            regime_detector: Trained RegimeDetector instance
            ensemble: Trained RegimeSpecializedEnsemble instance
            fallback_regime: Regime to use if detection fails (default: 1 = Complacency)
        """
        self.regime_detector = regime_detector
        self.ensemble = ensemble
        self.fallback_regime = fallback_regime
        
        # Validate that models are trained
        if not ensemble.models:
            raise ValueError("Ensemble has no trained models")
        
        logger.info(
            f"Initialized AdaptivePredictor with fallback_regime={fallback_regime}"
        )
    
    def _detect_regime(
        self,
        regime_features: np.ndarray
    ) -> Dict[str, Union[int, float, np.ndarray]]:
        """
        Detect the current market regime using the HMM.
        
        Args:
            regime_features: Features for regime detection
            
        Returns:
            Dictionary containing regime index, confidence, and probabilities
        """
        try:
            # Get regime prediction
            regime_pred = self.regime_detector.predict(regime_features)
            
            # Get regime probabilities if available
            try:
                regime_proba = self.regime_detector.predict_proba(regime_features)
                confidence = np.max(regime_proba, axis=1) if regime_proba.ndim > 1 else np.max(regime_proba)
            except (AttributeError, NotImplementedError):
                regime_proba = None
                confidence = 1.0
            
            return {
                'regime': regime_pred,
                'confidence': confidence,
                'probabilities': regime_proba
            }
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}. Using fallback regime.")
            n_samples = len(regime_features) if hasattr(regime_features, '__len__') else 1
            return {
                'regime': np.full(n_samples, self.fallback_regime),
                'confidence': np.zeros(n_samples),
                'probabilities': None
            }
    
    def predict(
        self,
        X: np.ndarray,
        regime_features: Optional[np.ndarray] = None,
        regime_override: Optional[int] = None
    ) -> Dict[str, Union[int, float, np.ndarray]]:
        """
        Make a prediction for a single sample or small batch.
        
        Args:
            X: Feature vector for quintile prediction (1D or 2D array)
            regime_features: Features for regime detection (if None, use X)
            regime_override: If provided, skip regime detection and use this regime
            
        Returns:
            Dictionary containing:
                - regime: Detected regime index
                - regime_confidence: Max probability from regime detection
                - quintile_prediction: Predicted quintile (1-5)
                - prediction_confidence: Max probability from quintile prediction
                - quintile_probabilities: Full probability array [P(Q1), ..., P(Q5)]
        """
        # Ensure X is 2D
        X = np.atleast_2d(X)
        
        if X.shape[0] != 1:
            logger.warning(
                f"predict() received {X.shape[0]} samples. "
                f"Use predict_batch() for multiple samples."
            )
        
        # Determine regime
        if regime_override is not None:
            regime = regime_override
            regime_confidence = 1.0
            regime_proba = None
        else:
            if regime_features is None:
                regime_features = X
            regime_features = np.atleast_2d(regime_features)
            
            regime_result = self._detect_regime(regime_features)
            regime = int(regime_result['regime'][0]) if isinstance(regime_result['regime'], np.ndarray) else regime_result['regime']
            regime_confidence = float(regime_result['confidence'][0]) if isinstance(regime_result['confidence'], np.ndarray) else regime_result['confidence']
            regime_proba = regime_result['probabilities']
        
        # Get the appropriate model
        model = self.ensemble.get_model(regime)
        if model is None:
            logger.warning(
                f"No model for regime {regime}, falling back to regime {self.fallback_regime}"
            )
            regime = self.fallback_regime
            model = self.ensemble.get_model(regime)
            if model is None:
                raise ValueError(f"No model available for fallback regime {self.fallback_regime}")
        
        # Make quintile prediction
        quintile_pred = self.ensemble.predict(X, regime)
        quintile_proba = self.ensemble.predict_proba(X, regime)
        
        # Extract single sample results
        quintile_prediction = int(quintile_pred[0])
        prediction_confidence = float(np.max(quintile_proba[0]))
        quintile_probabilities = quintile_proba[0].tolist()
        
        return {
            'regime': regime,
            'regime_confidence': regime_confidence,
            'quintile_prediction': quintile_prediction,
            'prediction_confidence': prediction_confidence,
            'quintile_probabilities': quintile_probabilities
        }
    
    def predict_batch(
        self,
        X: np.ndarray,
        regime_features: Optional[np.ndarray] = None,
        regime_override: Optional[Union[int, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions for a batch of samples efficiently.
        
        This method groups samples by detected regime and makes batch
        predictions for each group, which is more efficient than
        predicting one sample at a time.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            regime_features: Features for regime detection (if None, use X)
            regime_override: If provided, use these regime(s) instead of detecting
            
        Returns:
            Dictionary containing arrays:
                - regimes: Detected regime indices (n_samples,)
                - regime_confidences: Confidence scores (n_samples,)
                - quintile_predictions: Predicted quintiles 1-5 (n_samples,)
                - prediction_confidences: Max quintile probabilities (n_samples,)
                - quintile_probabilities: Full probability arrays (n_samples, 5)
        """
        X = np.atleast_2d(X)
        n_samples = X.shape[0]
        
        # Determine regimes for all samples
        if regime_override is not None:
            if isinstance(regime_override, int):
                regimes = np.full(n_samples, regime_override)
            else:
                regimes = np.array(regime_override)
            regime_confidences = np.ones(n_samples)
        else:
            if regime_features is None:
                regime_features = X
            regime_features = np.atleast_2d(regime_features)
            
            regime_result = self._detect_regime(regime_features)
            regimes = np.array(regime_result['regime']).flatten()
            regime_confidences = np.array(regime_result['confidence']).flatten()
        
        # Initialize output arrays
        quintile_predictions = np.zeros(n_samples, dtype=int)
        prediction_confidences = np.zeros(n_samples)
        quintile_probabilities = np.zeros((n_samples, 5))
        
        # Process each regime group
        for regime_idx in np.unique(regimes):
            mask = regimes == regime_idx
            X_group = X[mask]
            
            # Get model for this regime
            model = self.ensemble.get_model(int(regime_idx))
            if model is None:
                logger.warning(
                    f"No model for regime {regime_idx}, using fallback"
                )
                model = self.ensemble.get_model(self.fallback_regime)
                regimes[mask] = self.fallback_regime
            
            if model is not None:
                # Make predictions for this group
                preds = self.ensemble.predict(X_group, int(regime_idx))
                proba = self.ensemble.predict_proba(X_group, int(regime_idx))
                
                quintile_predictions[mask] = preds
                quintile_probabilities[mask] = proba
                prediction_confidences[mask] = np.max(proba, axis=1)
        
        return {
            'regimes': regimes,
            'regime_confidences': regime_confidences,
            'quintile_predictions': quintile_predictions,
            'prediction_confidences': prediction_confidences,
            'quintile_probabilities': quintile_probabilities
        }
    
    def get_regime_distribution(
        self,
        regime_features: np.ndarray
    ) -> Dict[str, float]:
        """
        Get the distribution of regime predictions for a dataset.
        
        Args:
            regime_features: Features for regime detection
            
        Returns:
            Dictionary mapping regime names to proportions
        """
        regime_result = self._detect_regime(regime_features)
        regimes = np.array(regime_result['regime']).flatten()
        
        regime_names = {0: 'Euphoria', 1: 'Complacency', 2: 'Capitulation'}
        distribution = {}
        
        for idx, name in regime_names.items():
            count = np.sum(regimes == idx)
            distribution[name] = count / len(regimes)
        
        return distribution
    
    def __repr__(self) -> str:
        """String representation of the predictor."""
        return (
            f"AdaptivePredictor("
            f"ensemble={self.ensemble}, "
            f"fallback_regime={self.fallback_regime})"
        )
