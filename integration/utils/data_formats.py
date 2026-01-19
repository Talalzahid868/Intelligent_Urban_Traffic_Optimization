"""
Unified data format definitions for the integration layer.

Provides standardized data structures used across all modules.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import numpy as np


@dataclass
class DetectionResult:
    """
    Result from vehicle/pedestrian detection (Vision Module).
    
    Attributes:
        image_name: Source image filename
        vehicle_count: Number of vehicles detected
        pedestrian_count: Number of pedestrians detected
        congestion: Congestion level (LOW, Medium, High)
        raw_detections: Optional list of individual detection boxes
        timestamp: When detection was performed
    """
    image_name: str
    vehicle_count: int
    pedestrian_count: int
    congestion: str
    raw_detections: Optional[List[dict]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            'image_name': self.image_name,
            'vehicle_count': self.vehicle_count,
            'pedestrian_count': self.pedestrian_count,
            'congestion': self.congestion,
            'timestamp': self.timestamp.isoformat()
        }
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for ML models."""
        return np.array([self.vehicle_count, self.pedestrian_count], dtype=np.float32)


@dataclass
class PredictionResult:
    """
    Result from traffic prediction (LSTM Module).
    
    Attributes:
        predicted_vehicle_count: Predicted number of vehicles
        predicted_pedestrian_count: Predicted number of pedestrians
        predicted_congestion: Predicted congestion level
        confidence: Model confidence in prediction
        horizon: Prediction time horizon (steps ahead)
    """
    predicted_vehicle_count: float
    predicted_pedestrian_count: float
    predicted_congestion: str
    confidence: float = 1.0
    horizon: int = 1
    
    def to_dict(self) -> dict:
        return {
            'predicted_vehicle_count': self.predicted_vehicle_count,
            'predicted_pedestrian_count': self.predicted_pedestrian_count,
            'predicted_congestion': self.predicted_congestion,
            'confidence': self.confidence,
            'horizon': self.horizon
        }


@dataclass
class AnomalyResult:
    """
    Result from anomaly detection (VAE Module).
    
    Attributes:
        reconstruction_error: MSE between input and reconstruction
        is_anomaly: Whether the input is classified as anomaly
        latent_representation: Encoded latent vector
        threshold: Anomaly detection threshold used
    """
    reconstruction_error: float
    is_anomaly: bool
    latent_representation: np.ndarray
    threshold: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'reconstruction_error': self.reconstruction_error,
            'is_anomaly': self.is_anomaly,
            'threshold': self.threshold
        }


@dataclass
class ActionResult:
    """
    Result from action selection (RL Module).
    
    Attributes:
        action_id: Numeric action identifier
        action_name: Human-readable action name
        expected_reward: Expected reward for this action
        state: Current state tuple
        q_values: Q-values for all actions
    """
    action_id: int
    action_name: str
    expected_reward: float
    state: Tuple[int, int]
    q_values: Optional[Dict[int, float]] = None
    
    def to_dict(self) -> dict:
        return {
            'action_id': self.action_id,
            'action_name': self.action_name,
            'expected_reward': self.expected_reward,
            'state': self.state
        }


@dataclass
class GenerationResult:
    """
    Result from synthetic data generation (GAN Module).
    
    Attributes:
        synthetic_data: Generated data array
        num_samples: Number of samples generated
    """
    synthetic_data: np.ndarray
    num_samples: int
    
    def to_dict(self) -> dict:
        return {
            'num_samples': self.num_samples,
            'data_shape': self.synthetic_data.shape
        }


@dataclass
class PipelineResult:
    """
    Complete result from running the full pipeline.
    
    Attributes:
        detection: Vision detection results
        prediction: LSTM prediction results
        anomaly: VAE anomaly detection results
        action: RL action recommendations
        processing_time: Total processing time in seconds
    """
    detection: Optional[DetectionResult] = None
    prediction: Optional[PredictionResult] = None
    anomaly: Optional[AnomalyResult] = None
    action: Optional[ActionResult] = None
    processing_time: float = 0.0
    
    def to_dict(self) -> dict:
        result = {'processing_time': self.processing_time}
        
        if self.detection:
            result['detection'] = self.detection.to_dict()
        if self.prediction:
            result['prediction'] = self.prediction.to_dict()
        if self.anomaly:
            result['anomaly'] = self.anomaly.to_dict()
        if self.action:
            result['action'] = self.action.to_dict()
        
        return result
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = ["Pipeline Result Summary", "=" * 40]
        
        if self.detection:
            lines.append(f"Detection: {self.detection.vehicle_count} vehicles, "
                        f"{self.detection.pedestrian_count} pedestrians, "
                        f"Congestion: {self.detection.congestion}")
        
        if self.prediction:
            lines.append(f"Prediction: {self.prediction.predicted_vehicle_count:.1f} vehicles, "
                        f"Congestion: {self.prediction.predicted_congestion}")
        
        if self.anomaly:
            status = "ANOMALY" if self.anomaly.is_anomaly else "Normal"
            lines.append(f"Anomaly Status: {status} "
                        f"(error: {self.anomaly.reconstruction_error:.4f})")
        
        if self.action:
            lines.append(f"Recommended Action: {self.action.action_name}")
        
        lines.append(f"Processing Time: {self.processing_time:.3f}s")
        
        return "\n".join(lines)
