"""
FID Monitor — GPU-accelerated real-time FID computation for GAN training.
Runs in a background thread with minimal interference to training.
"""

import threading
import queue
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np

# Import gan_monitor for socket.io emission
try:
    from src import gan_monitor
    HAS_MONITOR = True
except ImportError:
    HAS_MONITOR = False


class InceptionFeatureExtractor(nn.Module):
    """InceptionV3 truncated at pool3 layer for 2048-dim features."""
    
    def __init__(self, device='cuda'):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        
        # Build truncated model up to avgpool
        self.conv1 = inception.Conv2d_1a_3x3
        self.conv2a = inception.Conv2d_2a_3x3
        self.conv2b = inception.Conv2d_2b_3x3
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.conv3 = inception.Conv2d_3b_1x1
        self.conv4 = inception.Conv2d_4a_3x3
        self.maxpool2 = nn.MaxPool2d(3, stride=2)
        self.mixed5b = inception.Mixed_5b
        self.mixed5c = inception.Mixed_5c
        self.mixed5d = inception.Mixed_5d
        self.mixed6a = inception.Mixed_6a
        self.mixed6b = inception.Mixed_6b
        self.mixed6c = inception.Mixed_6c
        self.mixed6d = inception.Mixed_6d
        self.mixed6e = inception.Mixed_6e
        self.mixed7a = inception.Mixed_7a
        self.mixed7b = inception.Mixed_7b
        self.mixed7c = inception.Mixed_7c
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.to(device)
        self.eval()
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Expects input in range [0, 1], shape (B, C, H, W)
        # Resize to 299x299 if needed
        if x.shape[-1] != 299 or x.shape[-2] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Convert grayscale to RGB if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Normalize to Inception expected range [-1, 1]
        x = 2 * x - 1
        
        # Forward through truncated network
        x = self.conv1(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.mixed5b(x)
        x = self.mixed5c(x)
        x = self.mixed5d(x)
        x = self.mixed6a(x)
        x = self.mixed6b(x)
        x = self.mixed6c(x)
        x = self.mixed6d(x)
        x = self.mixed6e(x)
        x = self.mixed7a(x)
        x = self.mixed7b(x)
        x = self.mixed7c(x)
        x = self.avgpool(x)
        
        return x.view(x.size(0), -1)  # (B, 2048)


def _gpu_sqrtm(matrix):
    """
    Compute matrix square root using eigendecomposition on GPU.
    Σ^(1/2) = V @ diag(sqrt(λ)) @ V^T
    """
    # Use double precision for numerical stability
    matrix = matrix.double()
    
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    
    # Clamp small/negative eigenvalues (numerical stability)
    eigenvalues = torch.clamp(eigenvalues, min=0)
    
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    sqrt_matrix = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
    
    return sqrt_matrix.float()


def _compute_fid_from_stats(mu1, sigma1, mu2, sigma2):
    """
    Compute FID given precomputed statistics (all on GPU).
    FID = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2*(Σ1*Σ2)^(1/2))
    """
    diff = mu1 - mu2
    
    # Compute sqrt(Σ1 @ Σ2)
    covmean = _gpu_sqrtm(sigma1 @ sigma2)
    
    # Handle potential imaginary component from numerical errors
    if torch.is_complex(covmean):
        covmean = covmean.real
    
    fid = torch.sum(diff ** 2) + torch.trace(sigma1 + sigma2 - 2 * covmean)
    
    return fid.item()


def _compute_statistics(features):
    """Compute mean and covariance of features (on GPU)."""
    mu = torch.mean(features, dim=0)
    # Center features
    features_centered = features - mu.unsqueeze(0)
    # Covariance (with Bessel's correction)
    sigma = (features_centered.T @ features_centered) / (features.shape[0] - 1)
    return mu, sigma


class FIDMonitor:
    """
    Real-time FID monitor for GAN training.
    
    Usage:
        fid_monitor = FIDMonitor(real_loader, device='cuda')
        
        for step in range(num_steps):
            # ... training ...
            if step % 1000 == 0:
                fid_monitor.submit(generator, num_classes, latent_dim, step)
    """
    
    def __init__(self, real_loader, device='cuda', n_samples=1000, batch_size=100):
        """
        Args:
            real_loader: DataLoader for real images (used to compute reference stats)
            device: CUDA device
            n_samples: Number of samples for mini-FID computation
            batch_size: Batch size for generation/feature extraction
        """
        self.device = device
        self.n_samples = n_samples
        self.batch_size = batch_size
        
        # Create separate CUDA stream for FID computation
        self.fid_stream = torch.cuda.Stream(device=device)
        
        # Feature extractor (shared, inference only)
        print("[FIDMonitor] Loading InceptionV3...")
        self.inception = InceptionFeatureExtractor(device=device)
        
        # Precompute real statistics
        print(f"[FIDMonitor] Computing real statistics from {n_samples} samples...")
        self.mu_real, self.sigma_real = self._compute_real_stats(real_loader)
        print("[FIDMonitor] Ready.")
        
        # Background thread setup
        self._queue = queue.Queue(maxsize=2)  # Don't queue too many requests
        self._thread = None
        self._running = False
        
        # Latest FID value
        self.latest_fid = None
        self.latest_step = None
    
    def _compute_real_stats(self, loader):
        """Compute Inception statistics for real dataset."""
        features_list = []
        n_collected = 0
        
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                
                images = images.to(self.device)
                
                # Convert from [-1, 1] to [0, 1] if needed
                if images.min() < 0:
                    images = (images + 1) / 2
                
                features = self.inception(images)
                features_list.append(features)
                
                n_collected += images.shape[0]
                if n_collected >= self.n_samples:
                    break
        
        features = torch.cat(features_list, dim=0)[:self.n_samples]
        return _compute_statistics(features)
    
    def _worker(self):
        """Background worker thread."""
        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
                if item is None:  # Shutdown signal
                    break
                
                generator_state, num_classes, latent_dim, step = item
                
                # Run FID computation on separate stream
                with torch.cuda.stream(self.fid_stream):
                    fid = self._compute_fid(generator_state, num_classes, latent_dim)
                
                # Wait for computation to complete
                self.fid_stream.synchronize()
                
                self.latest_fid = fid
                self.latest_step = step
                
                print(f"[FIDMonitor] Step {step}: FID = {fid:.2f}")
                
                # Emit via socket.io if available
                if HAS_MONITOR:
                    gan_monitor._tx_queue.put(("fid_update", {
                        "step": step,
                        "fid": round(fid, 2)
                    }))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[FIDMonitor] Error: {e}")
    
    def _compute_fid(self, generator_state, num_classes, latent_dim):
        """Compute FID using cloned generator state."""
        features_list = []
        n_generated = 0
        
        # We need to reconstruct a generator - this requires knowing the architecture
        # Instead, we receive the generator's state_dict and a reference to create one
        # For simplicity, we'll use the generator directly but with torch.no_grad()
        
        # Actually, generator_state here is the generator module itself (cloned weights)
        generator = generator_state
        generator.eval()
        
        with torch.no_grad():
            while n_generated < self.n_samples:
                batch_size = min(self.batch_size, self.n_samples - n_generated)
                
                # Generate random latent vectors and labels
                z = torch.randn(batch_size, latent_dim, device=self.device)
                labels = torch.randint(0, num_classes, (batch_size,), device=self.device)
                
                # Generate images
                fake_images = generator(z, labels)
                
                # Convert from [-1, 1] to [0, 1]
                fake_images = (fake_images + 1) / 2
                
                # Extract features
                features = self.inception(fake_images)
                features_list.append(features)
                
                n_generated += batch_size
        
        features = torch.cat(features_list, dim=0)
        mu_fake, sigma_fake = _compute_statistics(features)
        
        return _compute_fid_from_stats(self.mu_real, self.sigma_real, mu_fake, sigma_fake)
    
    def start(self):
        """Start the background worker thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the background worker thread."""
        self._running = False
        self._queue.put(None)  # Shutdown signal
        if self._thread is not None:
            self._thread.join(timeout=2.0)
    
    def submit(self, generator, num_classes, latent_dim, step):
        """
        Submit a FID computation request (non-blocking).
        
        Args:
            generator: The generator model
            num_classes: Number of classes for conditional generation
            latent_dim: Dimension of latent vector
            step: Current training step
        """
        # Start worker if not running
        if not self._running:
            self.start()
        
        # Clone generator to avoid interference with training
        try:
            generator_clone = copy.deepcopy(generator)
            generator_clone.eval()
            
            # Try to queue, don't block if full
            self._queue.put_nowait((generator_clone, num_classes, latent_dim, step))
        except queue.Full:
            pass  # Skip this FID computation if queue is full
        except Exception as e:
            print(f"[FIDMonitor] Submit error: {e}")
    
    def get_latest(self):
        """Get the latest FID value and step."""
        return self.latest_fid, self.latest_step
