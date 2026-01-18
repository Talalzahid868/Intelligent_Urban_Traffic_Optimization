"""
LSTM Module Wrapper

Provides a standardized interface to the LSTM-based traffic prediction
module (module2_LSTM).
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class PredictionResult:
    """Result from LSTM traffic prediction."""
    predicted_vehicle_count: float
    predicted_pedestrian_count: float
    predicted_congestion: str
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            'predicted_vehicle_count': self.predicted_vehicle_count,
            'predicted_pedestrian_count': self.predicted_pedestrian_count,
            'predicted_congestion': self.predicted_congestion,
            'confidence': self.confidence
        }


class LSTMWrapper:
    """
    Wrapper for LSTM-based traffic prediction.
    
    This wrapper provides a clean interface to the LSTM model trained
    in module2_LSTM for predicting future traffic patterns.
    
    Example:
        wrapper = LSTMWrapper()
        sequence = np.array([...])  # shape: (sequence_length, 2)
        result = wrapper.predict(sequence)
        print(f"Predicted vehicles: {result.predicted_vehicle_count}")
    """
    
    def __init__(self, model_path: str = None, sequence_length: int = 10):
        """
        Initialize the LSTM wrapper.
        
        Args:
            model_path: Path to trained LSTM model (H5 format).
            sequence_length: Expected sequence length for input.
        """
        self._model = None
        self._model_path = model_path
        self.sequence_length = sequence_length
        self._feature_mean = None
        self._feature_std = None
    
    def _load_model(self):
        """Lazy load the LSTM model."""
        if self._model is None:
            try:
                from tensorflow.keras.models import load_model
                
                if self._model_path:
                    model_path = Path(self._model_path)
                else:
                    model_path = Path(__file__).parent.parent.parent / "module2_LSTM" / "Traffic_Model_LSTM.h5"
                
                if not model_path.exists():
                    raise FileNotFoundError(f"LSTM model not found: {model_path}")
                
                self._model = load_model(str(model_path))
                
            except ImportError:
                raise ImportError("tensorflow package required. Install with: pip install tensorflow")
    
    def set_normalization_params(self, mean: np.ndarray, std: np.ndarray):
        """
        Set normalization parameters for input/output transformation.
        
        Args:
            mean: Feature means from training data.
            std: Feature standard deviations from training data.
        """
        self._feature_mean = mean
        self._feature_std = std
    
    def predict(self, sequence: np.ndarray) -> PredictionResult:
        """
        Predict next traffic state from sequence.
        
        Args:
            sequence: Input sequence of shape (sequence_length, 2).
                     Features are [vehicle_count, pedestrian_count].
                     
        Returns:
            PredictionResult with predicted values.
        """
        self._load_model()
        
        # Ensure correct shape
        if sequence.ndim == 2:
            sequence = sequence.reshape(1, *sequence.shape)
        
        # Make prediction
        prediction = self._model.predict(sequence, verbose=0)
        
        # Denormalize if normalization params are set
        if self._feature_mean is not None:
            pred_denorm = prediction[0] * self._feature_std + self._feature_mean
        else:
            pred_denorm = prediction[0]
        
        vehicle_count = max(0, pred_denorm[0])
        pedestrian_count = max(0, pred_denorm[1]) if len(pred_denorm) > 1 else 0
        
        congestion = self._estimate_congestion(vehicle_count)
        
        return PredictionResult(
            predicted_vehicle_count=float(vehicle_count),
            predicted_pedestrian_count=float(pedestrian_count),
            predicted_congestion=congestion
        )
    
    def predict_multi_step(self, sequence: np.ndarray, steps: int) -> List[PredictionResult]:
        """
        Predict multiple future time steps.
        
        Args:
            sequence: Initial input sequence of shape (sequence_length, 2).
            steps: Number of future steps to predict.
            
        Returns:
            List of PredictionResults for each future step.
        """
        self._load_model()
        
        results = []
        current_seq = sequence.copy()
        
        for _ in range(steps):
            # Predict next step
            result = self.predict(current_seq)
            results.append(result)
            
            # Update sequence with prediction
            new_point = np.array([[
                result.predicted_vehicle_count,
                result.predicted_pedestrian_count
            ]])
            
            # Normalize if needed
            if self._feature_mean is not None:
                new_point = (new_point - self._feature_mean) / self._feature_std
            
            # Shift sequence and add new prediction
            current_seq = np.vstack([current_seq[1:], new_point])
        
        return results
    
    def predict_batch(self, sequences: np.ndarray) -> List[PredictionResult]:
        """
        Predict for a batch of sequences.
        
        Args:
            sequences: Batch of sequences, shape (batch_size, sequence_length, 2).
            
        Returns:
            List of PredictionResults.
        """
        self._load_model()
        
        predictions = self._model.predict(sequences, verbose=0)
        
        results = []
        for pred in predictions:
            if self._feature_mean is not None:
                pred_denorm = pred * self._feature_std + self._feature_mean
            else:
                pred_denorm = pred
            
            vehicle_count = max(0, pred_denorm[0])
            pedestrian_count = max(0, pred_denorm[1]) if len(pred_denorm) > 1 else 0
            
            results.append(PredictionResult(
                predicted_vehicle_count=float(vehicle_count),
                predicted_pedestrian_count=float(pedestrian_count),
                predicted_congestion=self._estimate_congestion(vehicle_count)
            ))
        
        return results
    
    @staticmethod
    def _estimate_congestion(vehicle_count: float) -> str:
        """Estimate congestion level from vehicle count."""
        if vehicle_count < 10:
            return "LOW"
        elif vehicle_count < 25:
            return "Medium"
        else:
            return "High"
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        self._load_model()
        import io
        stream = io.StringIO()
        self._model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()
