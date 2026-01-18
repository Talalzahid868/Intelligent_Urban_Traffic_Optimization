# Integration Layer for Intelligent Urban Traffic Optimization
# This package connects all deep learning modules into a unified pipeline

from .config import Config
from .data_pipeline import DataPipeline, TrafficData
from .pipeline import TrafficOptimizationPipeline

__all__ = [
    'Config',
    'DataPipeline', 
    'TrafficData',
    'TrafficOptimizationPipeline'
]

__version__ = '1.0.0'
