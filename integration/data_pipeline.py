"""
Unified data pipeline for inter-module communication.

Provides standardized data loading, preprocessing, and transformation
for all deep learning modules in the traffic optimization system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from datetime import datetime

from .config import Config, default_config


@dataclass
class TrafficData:
    """
    Unified data structure for traffic information.
    
    Attributes:
        image_name: Source image filename
        vehicle_count: Number of vehicles detected
        pedestrian_count: Number of pedestrians detected
        congestion: Congestion level (LOW, Medium, High)
        timestamp: When the data was captured/processed
        features: Normalized feature vector for ML models
    """
    image_name: str
    vehicle_count: int
    pedestrian_count: int
    congestion: str
    timestamp: datetime = field(default_factory=datetime.now)
    features: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'image_name': self.image_name,
            'vehicle_count': self.vehicle_count,
            'pedestrian_count': self.pedestrian_count,
            'congestion': self.congestion,
            'timestamp': self.timestamp.isoformat(),
        }
    
    def get_feature_vector(self) -> np.ndarray:
        """Get the raw feature vector [vehicle_count, pedestrian_count]."""
        return np.array([self.vehicle_count, self.pedestrian_count])
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TrafficData':
        """Create TrafficData from dictionary."""
        return cls(
            image_name=data['image_name'],
            vehicle_count=int(data['vehicle_count']),
            pedestrian_count=int(data['pedestrian_count']),
            congestion=data['congestion'],
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat()))
        )


class DataPipeline:
    """
    Unified data pipeline for the traffic optimization system.
    
    Handles data loading, preprocessing, normalization, and sequence
    generation for all modules.
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize the data pipeline.
        
        Args:
            config: Configuration object. Uses default if not provided.
        """
        self.config = config or default_config
        self._data: List[TrafficData] = []
        self._df: Optional[pd.DataFrame] = None
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
    
    def load_from_csv(self, filepath: Union[str, Path] = None) -> 'DataPipeline':
        """
        Load traffic data from CSV file.
        
        Args:
            filepath: Path to CSV file. Uses default module1 output if not provided.
            
        Returns:
            Self for method chaining.
        """
        if filepath is None:
            filepath = self.config.get_data_path("module1_results.csv")
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        self._df = pd.read_csv(filepath)
        self._data = [
            TrafficData(
                image_name=row['image_name'],
                vehicle_count=int(row['vehicle_count']),
                pedestrian_count=int(row['Pedestrian_count']),
                congestion=row['congestion']
            )
            for _, row in self._df.iterrows()
        ]
        
        return self
    
    def load_from_dataframe(self, df: pd.DataFrame) -> 'DataPipeline':
        """
        Load traffic data from a pandas DataFrame.
        
        Args:
            df: DataFrame with traffic data.
            
        Returns:
            Self for method chaining.
        """
        self._df = df.copy()
        self._data = [
            TrafficData(
                image_name=row['image_name'],
                vehicle_count=int(row['vehicle_count']),
                pedestrian_count=int(row.get('Pedestrian_count', row.get('pedestrian_count', 0))),
                congestion=row['congestion']
            )
            for _, row in self._df.iterrows()
        ]
        
        return self
    
    def get_feature_matrix(self, normalize: bool = None) -> np.ndarray:
        """
        Get feature matrix for ML models.
        
        Args:
            normalize: Whether to normalize features. Uses config default if not specified.
            
        Returns:
            Feature matrix of shape (n_samples, 2).
        """
        if normalize is None:
            normalize = self.config.pipeline.normalize_features
        
        features = np.array([
            [d.vehicle_count, d.pedestrian_count] 
            for d in self._data
        ], dtype=np.float32)
        
        if normalize:
            features = self._normalize_features(features)
        
        return features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using z-score normalization."""
        if self._feature_mean is None:
            self._feature_mean = features.mean(axis=0)
            self._feature_std = features.std(axis=0)
            # Avoid division by zero
            self._feature_std = np.where(self._feature_std == 0, 1, self._feature_std)
        
        return (features - self._feature_mean) / self._feature_std
    
    def denormalize_features(self, features: np.ndarray) -> np.ndarray:
        """Denormalize features back to original scale."""
        if self._feature_mean is None or self._feature_std is None:
            return features
        return features * self._feature_std + self._feature_mean
    
    def get_sequences(self, sequence_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sequences for LSTM model.
        
        Args:
            sequence_length: Length of each sequence. Uses config default if not specified.
            
        Returns:
            Tuple of (X, y) where X is sequences and y is targets.
        """
        if sequence_length is None:
            sequence_length = self.config.models.sequence_length
        
        features = self.get_feature_matrix(normalize=True)
        
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(features[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def get_congestion_labels(self) -> np.ndarray:
        """Get numeric congestion labels (0=LOW, 1=Medium, 2=High)."""
        label_map = {'LOW': 0, 'Medium': 1, 'High': 2}
        return np.array([label_map.get(d.congestion, 0) for d in self._data])
    
    def get_rl_states(self) -> np.ndarray:
        """
        Get discretized states for RL module.
        
        Returns:
            Array of state tuples (vehicle_level, pedestrian_level).
        """
        states = []
        for d in self._data:
            # Discretize vehicle count
            if d.vehicle_count < 10:
                v_level = 0  # low
            elif d.vehicle_count < 25:
                v_level = 1  # medium
            else:
                v_level = 2  # high
            
            # Discretize pedestrian count
            if d.pedestrian_count < 3:
                p_level = 0  # low
            elif d.pedestrian_count < 8:
                p_level = 1  # medium
            else:
                p_level = 2  # high
            
            states.append((v_level, p_level))
        
        return np.array(states)
    
    def add_synthetic_data(self, synthetic_features: np.ndarray) -> 'DataPipeline':
        """
        Add GAN-generated synthetic data to the pipeline.
        
        Args:
            synthetic_features: Synthetic feature matrix from GAN.
            
        Returns:
            Self for method chaining.
        """
        synthetic_features = self.denormalize_features(synthetic_features)
        
        for i, (v, p) in enumerate(synthetic_features):
            v_count = max(0, int(round(v)))
            p_count = max(0, int(round(p)))
            
            # Estimate congestion
            if v_count < self.config.pipeline.low_congestion_threshold:
                congestion = "LOW"
            elif v_count < self.config.pipeline.medium_congestion_threshold:
                congestion = "Medium"
            else:
                congestion = "High"
            
            self._data.append(TrafficData(
                image_name=f"synthetic_{i}.jpg",
                vehicle_count=v_count,
                pedestrian_count=p_count,
                congestion=congestion
            ))
        
        return self
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert pipeline data to pandas DataFrame."""
        return pd.DataFrame([d.to_dict() for d in self._data])
    
    def save_to_csv(self, filepath: Union[str, Path]) -> None:
        """Save pipeline data to CSV file."""
        self.to_dataframe().to_csv(filepath, index=False)
    
    def __len__(self) -> int:
        """Return number of data points."""
        return len(self._data)
    
    def __getitem__(self, idx: int) -> TrafficData:
        """Get data point by index."""
        return self._data[idx]
    
    def __iter__(self):
        """Iterate over data points."""
        return iter(self._data)


def estimate_congestion(vehicle_count: int, config: Config = None) -> str:
    """
    Estimate congestion level from vehicle count.
    
    Args:
        vehicle_count: Number of vehicles.
        config: Configuration with thresholds.
        
    Returns:
        Congestion level string.
    """
    if config is None:
        config = default_config
    
    if vehicle_count < config.pipeline.low_congestion_threshold:
        return "LOW"
    elif vehicle_count < config.pipeline.medium_congestion_threshold:
        return "Medium"
    else:
        return "High"
