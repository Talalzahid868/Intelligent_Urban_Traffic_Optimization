"""
Configuration management for the Traffic Optimization Integration Layer.

Provides centralized configuration for all module paths, model parameters,
and pipeline settings.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ModulePaths:
    """Paths to all module directories and their artifacts."""
    base_dir: Path
    
    @property
    def vision_dir(self) -> Path:
        return self.base_dir / "module1_vision"
    
    @property
    def lstm_dir(self) -> Path:
        return self.base_dir / "module2_LSTM"
    
    @property
    def vae_dir(self) -> Path:
        return self.base_dir / "module3_VEA"
    
    @property
    def gan_dir(self) -> Path:
        return self.base_dir / "module4_GAN"
    
    @property
    def rl_dir(self) -> Path:
        return self.base_dir / "module5_RL"
    
    @property
    def outputs_dir(self) -> Path:
        return self.base_dir / "Outputs"
    
    @property
    def archive_dir(self) -> Path:
        return self.base_dir / "archive"


@dataclass
class ModelConfig:
    """Configuration for pre-trained models."""
    # Vision Module (YOLOv8)
    yolo_model_path: str = "yolov8n.pt"
    vehicle_classes: tuple = (2, 3, 5, 7)  # car, motorcycle, bus, truck
    pedestrian_class: int = 0
    
    # LSTM Module
    lstm_model_path: str = "Traffic_Model_LSTM.h5"
    sequence_length: int = 10
    
    # VAE Module
    vae_latent_dim: int = 2
    
    # GAN Module
    gan_model_path: str = "traffic_generator.keras"
    gan_latent_dim: int = 8
    
    # RL Module
    rl_policy_path: str = "traffic_qlearning_policy.pkl"
    rl_actions: tuple = (
        "Short Green (10s)",
        "Medium Green (20s)", 
        "Long Green (30s)",
        "Emergency Mode"
    )


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""
    batch_size: int = 32
    
    # Congestion thresholds
    low_congestion_threshold: int = 10
    medium_congestion_threshold: int = 25
    
    # Data normalization
    normalize_features: bool = True
    
    # Output settings
    output_format: str = "csv"  # csv, json
    save_intermediate: bool = True


@dataclass 
class Config:
    """
    Main configuration class for the Traffic Optimization Integration Layer.
    
    Usage:
        config = Config()
        # or with custom base directory
        config = Config(base_dir=Path("/path/to/project"))
    """
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    paths: ModulePaths = field(init=False)
    models: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    def __post_init__(self):
        """Initialize derived configurations."""
        if isinstance(self.base_dir, str):
            self.base_dir = Path(self.base_dir)
        self.paths = ModulePaths(base_dir=self.base_dir)
        
        # Ensure output directory exists
        self.paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, module: str) -> Path:
        """Get the full path to a module's model file."""
        model_paths = {
            'vision': self.paths.vision_dir / self.models.yolo_model_path,
            'lstm': self.paths.lstm_dir / self.models.lstm_model_path,
            'gan': self.paths.gan_dir / self.models.gan_model_path,
            'rl': self.paths.rl_dir / self.models.rl_policy_path,
        }
        return model_paths.get(module)
    
    def get_data_path(self, filename: str) -> Path:
        """Get the full path to a data file in outputs directory."""
        return self.paths.outputs_dir / filename
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'base_dir': str(self.base_dir),
            'models': {
                'yolo_model_path': self.models.yolo_model_path,
                'lstm_model_path': self.models.lstm_model_path,
                'gan_model_path': self.models.gan_model_path,
                'rl_policy_path': self.models.rl_policy_path,
            },
            'pipeline': {
                'batch_size': self.pipeline.batch_size,
                'normalize_features': self.pipeline.normalize_features,
                'output_format': self.pipeline.output_format,
            }
        }
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        base_dir = os.environ.get('TRAFFIC_OPT_BASE_DIR')
        if base_dir:
            return cls(base_dir=Path(base_dir))
        return cls()


# Default configuration instance
default_config = Config()
