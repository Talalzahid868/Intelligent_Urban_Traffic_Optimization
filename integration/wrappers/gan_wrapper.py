"""
GAN Module Wrapper

Provides a standardized interface to the GAN-based synthetic traffic
data generation module (module4_GAN).
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers, Model


@dataclass
class GenerationResult:
    """Result from GAN data generation."""
    synthetic_data: np.ndarray
    num_samples: int
    
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(
            self.synthetic_data,
            columns=['vehicle_count', 'pedestrian_count']
        )


class GANWrapper:
    """
    Wrapper for GAN-based synthetic traffic data generation.
    
    This wrapper provides a clean interface to the GAN model trained
    in module4_GAN for generating synthetic traffic data samples.
    
    Example:
        wrapper = GANWrapper()
        wrapper.load_model("path/to/traffic_generator.keras")
        result = wrapper.generate(100)
        print(f"Generated {result.num_samples} samples")
    """
    
    def __init__(self, model_path: str = None, latent_dim: int = 8):
        """
        Initialize the GAN wrapper.
        
        Args:
            model_path: Path to trained generator model.
            latent_dim: Dimension of the latent noise vector.
        """
        self._generator = None
        self._model_path = model_path
        self.latent_dim = latent_dim
        self._feature_min = None
        self._feature_max = None
    
    def _load_model(self):
        """Lazy load the generator model."""
        if self._generator is None:
            if self._model_path:
                model_path = Path(self._model_path)
            else:
                model_path = Path(__file__).parent.parent.parent / "module4_GAN" / "traffic_generator.keras"
            
            if model_path.exists():
                try:
                    self._generator = tf.keras.models.load_model(str(model_path))
                except Exception as e:
                    print(f"Warning: Could not load model from {model_path}: {e}")
                    self._build_default_generator()
            else:
                print(f"Warning: Model not found at {model_path}, building default generator")
                self._build_default_generator()
    
    def _build_default_generator(self):
        """Build a default generator if model file not found."""
        self._generator = tf.keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(self.latent_dim,)),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu'),
            layers.Dense(2, activation='tanh')  # Output normalized [-1, 1]
        ])
    
    def set_data_range(self, feature_min: np.ndarray, feature_max: np.ndarray):
        """
        Set the data range for denormalization.
        
        Args:
            feature_min: Minimum values from training data.
            feature_max: Maximum values from training data.
        """
        self._feature_min = feature_min
        self._feature_max = feature_max
    
    def generate(self, num_samples: int, denormalize: bool = True) -> GenerationResult:
        """
        Generate synthetic traffic data samples.
        
        Args:
            num_samples: Number of samples to generate.
            denormalize: Whether to denormalize to original data range.
            
        Returns:
            GenerationResult with synthetic data.
        """
        self._load_model()
        
        # Generate random latent vectors
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        
        # Generate samples
        synthetic = self._generator.predict(noise, verbose=0)
        
        # Denormalize if range is set
        if denormalize and self._feature_min is not None:
            # Assumes output is in [-1, 1] range (tanh activation)
            synthetic = (synthetic + 1) / 2  # Scale to [0, 1]
            synthetic = synthetic * (self._feature_max - self._feature_min) + self._feature_min
        
        return GenerationResult(
            synthetic_data=synthetic,
            num_samples=num_samples
        )
    
    def augment_dataset(self, real_data: np.ndarray, augmentation_factor: float = 0.5) -> np.ndarray:
        """
        Augment a dataset with synthetic samples.
        
        Args:
            real_data: Original dataset of shape (n_samples, n_features).
            augmentation_factor: Fraction of synthetic samples to add (0.5 = 50% more).
            
        Returns:
            Augmented dataset combining real and synthetic samples.
        """
        # Set data range from real data
        self._feature_min = real_data.min(axis=0)
        self._feature_max = real_data.max(axis=0)
        
        # Generate synthetic samples
        num_synthetic = int(len(real_data) * augmentation_factor)
        result = self.generate(num_synthetic, denormalize=True)
        
        # Combine datasets
        augmented = np.vstack([real_data, result.synthetic_data])
        
        return augmented
    
    def train(self, real_data: np.ndarray, epochs: int = 100, batch_size: int = 32, verbose: int = 0):
        """
        Train the GAN on real data (simplified training loop).
        
        Args:
            real_data: Real training data.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            verbose: Verbosity level.
        """
        # Store data range for denormalization
        self._feature_min = real_data.min(axis=0)
        self._feature_max = real_data.max(axis=0)
        
        # Normalize data to [-1, 1]
        data_normalized = 2 * (real_data - self._feature_min) / (self._feature_max - self._feature_min) - 1
        
        data_dim = real_data.shape[1]
        
        # Build generator
        self._generator = tf.keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(self.latent_dim,)),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu'),
            layers.Dense(data_dim, activation='tanh')
        ])
        
        # Build discriminator
        discriminator = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(data_dim,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile
        discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Combined model
        discriminator.trainable = False
        gan_input = layers.Input(shape=(self.latent_dim,))
        generated = self._generator(gan_input)
        gan_output = discriminator(generated)
        gan = Model(gan_input, gan_output)
        gan.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Training loop
        n_samples = len(data_normalized)
        
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, n_samples, batch_size)
            real_batch = data_normalized[idx]
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_batch = self._generator.predict(noise, verbose=0)
            
            d_loss_real = discriminator.train_on_batch(real_batch, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(fake_batch, np.zeros((batch_size, 1)))
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: D_loss={d_loss_real[0]:.4f}/{d_loss_fake[0]:.4f}, G_loss={g_loss:.4f}")
    
    def save_generator(self, filepath: str):
        """Save the generator model."""
        if self._generator:
            self._generator.save(filepath)
    
    @property
    def is_loaded(self) -> bool:
        """Check if generator is loaded."""
        return self._generator is not None
