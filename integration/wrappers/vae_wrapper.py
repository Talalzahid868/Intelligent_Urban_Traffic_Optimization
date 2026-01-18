"""
VAE Module Wrapper

Provides a standardized interface to the Variational Autoencoder (VAE)
module (module3_VEA) for feature analysis and anomaly detection.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model


@dataclass
class AnomalyResult:
    """Result from VAE anomaly detection."""
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


class VAEModel(Model):
    """
    Variational Autoencoder model for traffic data.
    
    Reconstructed from module3_VEA/VAE.ipynb implementation.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 2):
        super(VAEModel, self).__init__()
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu'),
        ])
        
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(8, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(input_dim)
        ])
    
    def sample(self, z_mean, z_log_var):
        """Reparameterization trick."""
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps
    
    def encode(self, x):
        """Encode input to latent space."""
        h = self.encoder(x)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        return z_mean, z_log_var
    
    def decode(self, z):
        """Decode from latent space."""
        return self.decoder(z)
    
    def call(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.sample(z_mean, z_log_var)
        reconstruction = self.decode(z)
        
        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.add_loss(kl_loss)
        
        return reconstruction


class VAEWrapper:
    """
    Wrapper for VAE-based feature analysis and anomaly detection.
    
    This wrapper provides a clean interface to the VAE model trained
    in module3_VEA for analyzing traffic patterns and detecting anomalies.
    
    Example:
        wrapper = VAEWrapper()
        wrapper.fit(training_features)
        anomaly = wrapper.detect_anomalies(test_features)
        print(f"Is anomaly: {anomaly.is_anomaly}")
    """
    
    def __init__(self, latent_dim: int = 2, anomaly_threshold: float = None):
        """
        Initialize the VAE wrapper.
        
        Args:
            latent_dim: Dimension of the latent space.
            anomaly_threshold: Threshold for anomaly detection. 
                             Auto-computed if not provided.
        """
        self._model: Optional[VAEModel] = None
        self.latent_dim = latent_dim
        self.anomaly_threshold = anomaly_threshold
        self._is_fitted = False
        self._feature_mean = None
        self._feature_std = None
    
    def _build_model(self, input_dim: int):
        """Build the VAE model."""
        self._model = VAEModel(input_dim=input_dim, latent_dim=self.latent_dim)
        self._model.compile(optimizer='adam', loss='mse')
    
    def fit(self, features: np.ndarray, epochs: int = 50, verbose: int = 0) -> 'VAEWrapper':
        """
        Train the VAE on feature data.
        
        Args:
            features: Training features of shape (n_samples, n_features).
            epochs: Number of training epochs.
            verbose: Verbosity level.
            
        Returns:
            Self for method chaining.
        """
        # Normalize features
        self._feature_mean = features.mean(axis=0)
        self._feature_std = features.std(axis=0)
        self._feature_std = np.where(self._feature_std == 0, 1, self._feature_std)
        
        features_norm = (features - self._feature_mean) / self._feature_std
        
        # Build and train model
        self._build_model(features.shape[1])
        self._model.fit(features_norm, features_norm, epochs=epochs, verbose=verbose)
        
        # Compute anomaly threshold if not set
        if self.anomaly_threshold is None:
            reconstructions = self._model.predict(features_norm, verbose=0)
            errors = np.mean((features_norm - reconstructions) ** 2, axis=1)
            self.anomaly_threshold = np.mean(errors) + 2 * np.std(errors)
        
        self._is_fitted = True
        return self
    
    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encode features to latent space.
        
        Args:
            features: Input features of shape (n_samples, n_features).
            
        Returns:
            Latent representations of shape (n_samples, latent_dim).
        """
        if not self._is_fitted:
            raise RuntimeError("VAE must be fitted before encoding.")
        
        features_norm = (features - self._feature_mean) / self._feature_std
        z_mean, _ = self._model.encode(features_norm)
        return z_mean.numpy()
    
    def decode(self, latent: np.ndarray) -> np.ndarray:
        """
        Decode from latent space.
        
        Args:
            latent: Latent vectors of shape (n_samples, latent_dim).
            
        Returns:
            Reconstructed features of shape (n_samples, n_features).
        """
        if not self._is_fitted:
            raise RuntimeError("VAE must be fitted before decoding.")
        
        decoded_norm = self._model.decode(latent).numpy()
        return decoded_norm * self._feature_std + self._feature_mean
    
    def reconstruct(self, features: np.ndarray) -> np.ndarray:
        """
        Reconstruct features through the VAE.
        
        Args:
            features: Input features.
            
        Returns:
            Reconstructed features.
        """
        if not self._is_fitted:
            raise RuntimeError("VAE must be fitted before reconstruction.")
        
        features_norm = (features - self._feature_mean) / self._feature_std
        reconstructed_norm = self._model.predict(features_norm, verbose=0)
        return reconstructed_norm * self._feature_std + self._feature_mean
    
    def detect_anomalies(self, features: np.ndarray) -> AnomalyResult:
        """
        Detect anomalies in traffic data.
        
        Args:
            features: Input features to analyze.
            
        Returns:
            AnomalyResult with detection information.
        """
        if not self._is_fitted:
            raise RuntimeError("VAE must be fitted before anomaly detection.")
        
        features_norm = (features - self._feature_mean) / self._feature_std
        
        # Handle single sample
        if features_norm.ndim == 1:
            features_norm = features_norm.reshape(1, -1)
        
        # Get reconstruction
        reconstructed = self._model.predict(features_norm, verbose=0)
        
        # Compute reconstruction error
        error = np.mean((features_norm - reconstructed) ** 2, axis=1)
        mean_error = float(np.mean(error))
        
        # Get latent representation
        z_mean, _ = self._model.encode(features_norm)
        
        return AnomalyResult(
            reconstruction_error=mean_error,
            is_anomaly=mean_error > self.anomaly_threshold,
            latent_representation=z_mean.numpy(),
            threshold=self.anomaly_threshold
        )
    
    def detect_anomalies_batch(self, features: np.ndarray) -> list:
        """
        Detect anomalies for a batch of samples.
        
        Args:
            features: Batch of features.
            
        Returns:
            List of AnomalyResults.
        """
        if not self._is_fitted:
            raise RuntimeError("VAE must be fitted before anomaly detection.")
        
        features_norm = (features - self._feature_mean) / self._feature_std
        reconstructed = self._model.predict(features_norm, verbose=0)
        errors = np.mean((features_norm - reconstructed) ** 2, axis=1)
        z_mean, _ = self._model.encode(features_norm)
        
        results = []
        for i, error in enumerate(errors):
            results.append(AnomalyResult(
                reconstruction_error=float(error),
                is_anomaly=error > self.anomaly_threshold,
                latent_representation=z_mean[i].numpy(),
                threshold=self.anomaly_threshold
            ))
        
        return results
    
    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted
