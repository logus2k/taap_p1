# Conditional GAN Loss Strategy Comparison on MNIST

A comparative study of different loss functions for training conditional Generative Adversarial Networks on the MNIST dataset.

**Course:** Advanced Topics in Deep Learning  
**Topic:** Generative Adversarial Networks (GANs)

## Overview

This project implements and benchmarks four GAN loss strategies to evaluate their impact on image generation quality:

- **BCE** — Binary Cross-Entropy (original GAN formulation)
- **LSGAN** — Least Squares GAN
- **Hinge** — Hinge loss (used in SAGAN, BigGAN)
- **WGAN-GP** — Wasserstein GAN with Gradient Penalty

All strategies use the same Generator and Discriminator architectures, training hyperparameters, and evaluation metrics, enabling fair comparison.

## Architecture

### Generator
- **Input:** 100-dimensional latent vector + class embedding (10 classes)
- **Upsampling:** PixelShuffle blocks with ICNR initialization
- **Normalization:** BatchNorm with momentum 0.8
- **Output:** 28×28 grayscale image (tanh activation)

### Discriminator
- **Input:** 28×28 image concatenated with class embedding
- **Regularization:** Spectral Normalization on all layers
- **Architecture:** Convolutional blocks with LeakyReLU
- **Output:** Probability score (sigmoid for BCE/LSGAN, linear for Hinge/WGAN-GP)

### Training Techniques
- **TTUR:** Two Time-Scale Update Rule (LR_D=4e-4, LR_G=1e-4)
- **Label Smoothing:** 0.9 for real labels (BCE/LSGAN only)
- **Gradient Penalty:** λ=10 (WGAN-GP only)
- **Critic Steps:** 5 per generator step (WGAN-GP only)

## Evaluation Metrics

- **FID** (Fréchet Inception Distance) — measures distribution similarity
- **KID** (Kernel Inception Distance) — unbiased alternative to FID
- **Training Time** — wall-clock time comparison

## Project Structure

```
├── draft_01_v3.ipynb      # Main notebook with training and evaluation
├── bin/
│   ├── gan_monitor.py     # Live training monitor server
│   └── run_game.py        # Interactive game server
├── web/
│   ├── index.html         # Training monitor dashboard
│   └── game.html          # Human vs GAN game interface
├── model/
│   ├── G_bce.pt           # Generator weights (per strategy)
│   └── D_bce.pt           # Discriminator weights (per strategy)
└── README.md
```

## Usage

### Training & Benchmarking

Run the Jupyter notebook `draft_01_v3.ipynb` to:
1. Train all four loss strategies
2. Compute FID/KID metrics
3. Generate comparison visualizations
4. Save model checkpoints

### Live Training Monitor (Optional)

Enable real-time monitoring by setting `LIVE_MONITOR = True` in the notebook, then open `http://localhost:8992` to view:
- Loss curves (Generator and Discriminator)
- Generated samples during training
- Benchmark progress across strategies

### Interactive Demo (Optional)

After training, run the game server to compare human drawings against the Generator:

```bash
python bin/run_game.py --strategy lsgan --port 8993
```

## Requirements

```
torch>=2.0
torchvision
torchmetrics
numpy
matplotlib
python-socketio (optional, for live monitor)
uvicorn (optional, for live monitor)
fastapi (optional, for live monitor)
```

## Results

The notebook produces:
- FID/KID comparison bar charts
- Loss curves for all strategies
- Per-class generation quality analysis
- Side-by-side sample comparisons
- CSV export of benchmark results

## License

Apache License 2.0
