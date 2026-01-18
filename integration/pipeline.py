"""
Traffic Optimization Pipeline Orchestrator

Main orchestration layer that coordinates all deep learning modules
for end-to-end traffic analysis and optimization.
"""

import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union

from .config import Config, default_config
from .data_pipeline import DataPipeline, TrafficData
from .utils.logging_utils import PipelineLogger, log_module_status
from .utils.data_formats import PipelineResult


@dataclass
class PipelineOptions:
    """Options for pipeline execution."""
    enable_vision: bool = True
    enable_lstm: bool = True
    enable_vae: bool = True
    enable_gan: bool = False  # Off by default as it modifies data
    enable_rl: bool = True
    enable_nlp: bool = False  # Off by default, requires social media data
    
    # VAE training options (only if VAE model not pre-trained)
    vae_epochs: int = 50
    
    # LSTM prediction options
    lstm_prediction_steps: int = 1
    
    # GAN augmentation options
    gan_augmentation_factor: float = 0.5


class TrafficOptimizationPipeline:
    """
    Main orchestrator for the traffic optimization pipeline.
    
    Coordinates all deep learning modules to process traffic data
    and provide optimization recommendations.
    
    Example:
        pipeline = TrafficOptimizationPipeline()
        pipeline.load_data("data.csv")
        results = pipeline.run()
        print(results.summary())
    """
    
    def __init__(self, config: Config = None, options: PipelineOptions = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration object. Uses default if not provided.
            options: Pipeline execution options.
        """
        self.config = config or default_config
        self.options = options or PipelineOptions()
        self.logger = PipelineLogger("orchestrator")
        
        # Module wrappers (lazy loaded)
        self._vision_wrapper = None
        self._lstm_wrapper = None
        self._vae_wrapper = None
        self._gan_wrapper = None
        self._rl_wrapper = None
        self._nlp_wrapper = None
        
        # Data pipeline
        self.data_pipeline: Optional[DataPipeline] = None
        
        # Results cache
        self._results: List[PipelineResult] = []
    
    def _load_vision(self):
        """Lazy load vision module."""
        if self._vision_wrapper is None and self.options.enable_vision:
            try:
                from .wrappers import VisionWrapper
                self._vision_wrapper = VisionWrapper(
                    model_path=str(self.config.get_model_path('vision'))
                )
                log_module_status("Vision", "loaded")
            except Exception as e:
                log_module_status("Vision", "error", str(e))
    
    def _load_lstm(self):
        """Lazy load LSTM module."""
        if self._lstm_wrapper is None and self.options.enable_lstm:
            try:
                from .wrappers import LSTMWrapper
                self._lstm_wrapper = LSTMWrapper(
                    model_path=str(self.config.get_model_path('lstm')),
                    sequence_length=self.config.models.sequence_length
                )
                log_module_status("LSTM", "loaded")
            except Exception as e:
                log_module_status("LSTM", "error", str(e))
    
    def _load_vae(self):
        """Lazy load VAE module."""
        if self._vae_wrapper is None and self.options.enable_vae:
            try:
                from .wrappers import VAEWrapper
                self._vae_wrapper = VAEWrapper(
                    latent_dim=self.config.models.vae_latent_dim
                )
                log_module_status("VAE", "loaded")
            except Exception as e:
                log_module_status("VAE", "error", str(e))
    
    def _load_gan(self):
        """Lazy load GAN module."""
        if self._gan_wrapper is None and self.options.enable_gan:
            try:
                from .wrappers import GANWrapper
                self._gan_wrapper = GANWrapper(
                    model_path=str(self.config.get_model_path('gan')),
                    latent_dim=self.config.models.gan_latent_dim
                )
                log_module_status("GAN", "loaded")
            except Exception as e:
                log_module_status("GAN", "error", str(e))
    
    def _load_rl(self):
        """Lazy load RL module."""
        if self._rl_wrapper is None and self.options.enable_rl:
            try:
                from .wrappers import RLWrapper
                self._rl_wrapper = RLWrapper(
                    policy_path=str(self.config.get_model_path('rl'))
                )
                log_module_status("RL", "loaded")
            except Exception as e:
                log_module_status("RL", "error", str(e))
    
    def _load_nlp(self):
        """Lazy load NLP module."""
        if self._nlp_wrapper is None and self.options.enable_nlp:
            try:
                from .wrappers import NLPWrapper
                self._nlp_wrapper = NLPWrapper(
                    device=self.config.models.nlp_device
                )
                log_module_status("NLP", "loaded")
            except Exception as e:
                log_module_status("NLP", "error", str(e))
    
    def load_all_modules(self):
        """Pre-load all enabled modules."""
        self.logger.start("Loading modules")
        
        if self.options.enable_vision:
            self._load_vision()
        if self.options.enable_lstm:
            self._load_lstm()
        if self.options.enable_vae:
            self._load_vae()
        if self.options.enable_gan:
            self._load_gan()
        if self.options.enable_rl:
            self._load_rl()
        if self.options.enable_nlp:
            self._load_nlp()
        
        self.logger.end("Loading modules")
    
    def load_data(self, filepath: Union[str, Path] = None) -> 'TrafficOptimizationPipeline':
        """
        Load data into the pipeline.
        
        Args:
            filepath: Path to CSV data file. Uses default if not provided.
            
        Returns:
            Self for method chaining.
        """
        self.logger.start("Loading data")
        
        self.data_pipeline = DataPipeline(self.config)
        self.data_pipeline.load_from_csv(filepath)
        
        self.logger.step(f"Loaded {len(self.data_pipeline)} samples")
        self.logger.end("Loading data")
        
        return self
    
    def run_vision(self, image: np.ndarray, image_name: str = "input.jpg"):
        """
        Run vision detection on a single image.
        
        Args:
            image: Input image as numpy array.
            image_name: Name/identifier for the image.
            
        Returns:
            DetectionResult from vision module.
        """
        self._load_vision()
        
        if self._vision_wrapper is None:
            self.logger.warning("Vision module not available")
            return None
        
        return self._vision_wrapper.detect(image, image_name)
    
    def run_prediction(self, sequence: np.ndarray = None):
        """
        Run traffic prediction on sequence data.
        
        Args:
            sequence: Input sequence. Uses pipeline data if not provided.
            
        Returns:
            PredictionResult or list of results.
        """
        self._load_lstm()
        
        if self._lstm_wrapper is None:
            self.logger.warning("LSTM module not available")
            return None
        
        if sequence is None and self.data_pipeline is not None:
            X, y = self.data_pipeline.get_sequences()
            if len(X) > 0:
                # Set normalization params
                features = self.data_pipeline.get_feature_matrix(normalize=False)
                mean = features.mean(axis=0)
                std = features.std(axis=0)
                self._lstm_wrapper.set_normalization_params(mean, std)
                
                # Predict on last sequence
                return self._lstm_wrapper.predict(X[-1])
        elif sequence is not None:
            return self._lstm_wrapper.predict(sequence)
        
        return None
    
    def run_anomaly_detection(self, features: np.ndarray = None):
        """
        Run anomaly detection on features.
        
        Args:
            features: Input features. Uses pipeline data if not provided.
            
        Returns:
            AnomalyResult or list of results.
        """
        self._load_vae()
        
        if self._vae_wrapper is None:
            self.logger.warning("VAE module not available")
            return None
        
        if features is None and self.data_pipeline is not None:
            features = self.data_pipeline.get_feature_matrix(normalize=False)
        
        if features is None:
            return None
        
        # Fit VAE if not already fitted
        if not self._vae_wrapper.is_fitted:
            self.logger.step("Training VAE on data...")
            self._vae_wrapper.fit(features, epochs=self.options.vae_epochs)
        
        # Detect anomalies on last sample
        return self._vae_wrapper.detect_anomalies(features[-1:])
    
    def run_data_augmentation(self, features: np.ndarray = None):
        """
        Run GAN data augmentation.
        
        Args:
            features: Data to augment. Uses pipeline data if not provided.
            
        Returns:
            Augmented data array.
        """
        self._load_gan()
        
        if self._gan_wrapper is None:
            self.logger.warning("GAN module not available")
            return None
        
        if features is None and self.data_pipeline is not None:
            features = self.data_pipeline.get_feature_matrix(normalize=False)
        
        if features is None:
            return None
        
        return self._gan_wrapper.augment_dataset(
            features, 
            augmentation_factor=self.options.gan_augmentation_factor
        )
    
    def run_optimization(self, vehicle_count: int = None, pedestrian_count: int = None):
        """
        Get traffic signal optimization recommendation.
        
        Args:
            vehicle_count: Current vehicle count.
            pedestrian_count: Current pedestrian count.
            
        Returns:
            ActionResult with recommendation.
        """
        self._load_rl()
        
        if self._rl_wrapper is None:
            self.logger.warning("RL module not available")
            return None
        
        # Use last data point if counts not provided
        if vehicle_count is None and self.data_pipeline is not None:
            last_data = self.data_pipeline[-1]
            vehicle_count = last_data.vehicle_count
            pedestrian_count = last_data.pedestrian_count
        
        if vehicle_count is None:
            return None
        
        return self._rl_wrapper.get_action(
            vehicle_count=vehicle_count,
            pedestrian_count=pedestrian_count or 0
        )
    
    def run_nlp_analysis(self, text: str = None, texts: list = None):
        """
        Run NLP analysis on social media text.
        
        Args:
            text: Single text to analyze.
            texts: List of texts to analyze in batch.
            
        Returns:
            NLPAnalysisResult or list of results.
        """
        self._load_nlp()
        
        if self._nlp_wrapper is None:
            self.logger.warning("NLP module not available")
            return None
        
        if texts is not None:
            return self._nlp_wrapper.analyze_batch(texts)
        elif text is not None:
            return self._nlp_wrapper.analyze_text(text)
        
        return None
    
    def run(self) -> PipelineResult:
        """
        Run the complete pipeline on loaded data.
        
        Returns:
            PipelineResult with all outputs.
        """
        self.logger.start("Running pipeline")
        start_time = time.time()
        
        result = PipelineResult()
        
        if self.data_pipeline is None or len(self.data_pipeline) == 0:
            self.logger.error("No data loaded. Call load_data() first.")
            return result
        
        # Get latest data point for analysis
        latest = self.data_pipeline[-1]
        
        # Run LSTM prediction
        if self.options.enable_lstm:
            self.logger.step("Running traffic prediction...")
            try:
                prediction = self.run_prediction()
                if prediction:
                    result.prediction = prediction
                    self.logger.result("Prediction", 
                        f"{prediction.predicted_vehicle_count:.1f} vehicles")
            except Exception as e:
                self.logger.error("LSTM prediction failed", e)
        
        # Run VAE anomaly detection  
        if self.options.enable_vae:
            self.logger.step("Running anomaly detection...")
            try:
                anomaly = self.run_anomaly_detection()
                if anomaly:
                    result.anomaly = anomaly
                    status = "ANOMALY" if anomaly.is_anomaly else "Normal"
                    self.logger.result("Anomaly Status", status)
            except Exception as e:
                self.logger.error("VAE anomaly detection failed", e)
        
        # Run GAN augmentation (if enabled)
        if self.options.enable_gan:
            self.logger.step("Running data augmentation...")
            try:
                augmented = self.run_data_augmentation()
                if augmented is not None:
                    self.logger.result("Augmented samples", len(augmented))
            except Exception as e:
                self.logger.error("GAN augmentation failed", e)
        
        # Run RL optimization
        if self.options.enable_rl:
            self.logger.step("Getting signal optimization...")
            try:
                action = self.run_optimization(
                    latest.vehicle_count, 
                    latest.pedestrian_count
                )
                if action:
                    result.action = action
                    self.logger.result("Recommended Action", action.action_name)
            except Exception as e:
                self.logger.error("RL optimization failed", e)
        
        result.processing_time = time.time() - start_time
        self._results.append(result)
        
        self.logger.end("Running pipeline")
        
        return result
    
    def run_batch(self) -> List[PipelineResult]:
        """
        Run pipeline on all data points.
        
        Returns:
            List of PipelineResults.
        """
        if self.data_pipeline is None:
            self.logger.error("No data loaded")
            return []
        
        self.logger.start(f"Running batch pipeline on {len(self.data_pipeline)} samples")
        
        results = []
        features = self.data_pipeline.get_feature_matrix(normalize=False)
        
        # Pre-fit VAE on all data
        if self.options.enable_vae:
            self._load_vae()
            if self._vae_wrapper and not self._vae_wrapper.is_fitted:
                self._vae_wrapper.fit(features, epochs=self.options.vae_epochs)
        
        # Process each sample
        for i, data in enumerate(self.data_pipeline):
            result = PipelineResult()
            
            # RL action
            if self.options.enable_rl and self._rl_wrapper:
                result.action = self._rl_wrapper.get_action(
                    vehicle_count=data.vehicle_count,
                    pedestrian_count=data.pedestrian_count
                )
            
            # Anomaly detection
            if self.options.enable_vae and self._vae_wrapper:
                feature = np.array([[data.vehicle_count, data.pedestrian_count]])
                result.anomaly = self._vae_wrapper.detect_anomalies(feature)
            
            results.append(result)
        
        self.logger.end(f"Batch pipeline ({len(results)} results)")
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of pipeline state and results."""
        return {
            'config': self.config.to_dict(),
            'data_loaded': self.data_pipeline is not None,
            'data_size': len(self.data_pipeline) if self.data_pipeline else 0,
            'modules_loaded': {
                'vision': self._vision_wrapper is not None,
                'lstm': self._lstm_wrapper is not None,
                'vae': self._vae_wrapper is not None and self._vae_wrapper.is_fitted,
                'gan': self._gan_wrapper is not None,
                'rl': self._rl_wrapper is not None,
                'nlp': self._nlp_wrapper is not None,
            },
            'results_count': len(self._results)
        }
