# Human vs GAN — Adversarial Challenge

A web-based game where humans compete against a Generative Adversarial Network (GAN) to draw MNIST-style digits. Both the human and the Generator create digits, and the Discriminator judges which looks more authentic.

![Tug of War](web/tug_of_war.png)

## Overview

This project demonstrates GAN concepts interactively:

- **Generator**: A neural network trained to produce realistic handwritten digits
- **Discriminator**: A neural network trained to distinguish real digits from generated ones
- **Human Player**: You! Draw digits and try to fool the Discriminator

Each round, a random digit (0-9) is selected. The Generator produces its version instantly, then you draw yours. The Discriminator scores both — highest score wins the round.

## Features

- Real-time digit drawing with MNIST-style preprocessing
- Live Generator output display
- Discriminator scoring for both human and AI drawings
- MNIST View toggle to see exactly what the model sees
- Cumulative score tracking across rounds
- Support for multiple loss strategies (BCE, LSGAN, Hinge, WGAN-GP)

## Architecture

### Models

**Generator** (PixelShuffle + ICNR initialization):
- Input: 100-dim latent vector + class embedding
- Architecture: FC → PixelShuffle upsampling blocks → 28×28 output
- Activation: Tanh (outputs in [-1, 1])

**Discriminator** (Spectral Normalization):
- Input: 28×28 image + class embedding
- Architecture: Conv blocks with spectral normalization
- Output: Probability score (0-1)

### Training Improvements

- TTUR (Two Time-Scale Update Rule): LR_D = 4e-4, LR_G = 1e-4
- Label smoothing (0.9 for real labels)
- Spectral normalization on Discriminator
- ICNR initialization for PixelShuffle layers

## Installation

### Requirements

```
torch>=2.0
numpy
socketio
uvicorn
fastapi
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/human-vs-gan.git
cd human-vs-gan
```

2. Install dependencies:
```bash
pip install torch numpy python-socketio uvicorn fastapi
```

3. Train models or use pre-trained weights:
```bash
# Models should be saved as:
# model/G_bce.pt (Generator)
# model/D_bce.pt (Discriminator)
```

## Usage

### Running the Game Server

```bash
python bin/run_game.py
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8993 | Server port |
| `--host` | 0.0.0.0 | Server host |
| `--model-dir` | ./model | Directory containing .pt files |
| `--strategy` | bce | Loss strategy (bce/lsgan/hinge/wgan-gp) |
| `--latent-dim` | 100 | Latent dimension (must match training) |
| `--cpu` | False | Force CPU mode |

Then open `http://localhost:8993` in your browser.

### nginx Configuration (Optional)

For production deployment behind nginx:

```nginx
location /gan_game/ {
    proxy_pass http://localhost:8993/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}

location /gan_game/socket.io/ {
    proxy_pass http://localhost:8993/socket.io/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

## Project Structure

```
human-vs-gan/
├── bin/
│   └── run_game.py       # Standalone game server
├── web/
│   ├── game.html         # Game UI
│   └── tug_of_war.png    # Header image
├── model/
│   ├── G_bce.pt          # Generator weights
│   └── D_bce.pt          # Discriminator weights
├── README.md
└── LICENSE
```

## How It Works

1. **Round Start**: Random digit selected, Generator produces its version
2. **Drawing**: Human draws their interpretation of the digit
3. **Preprocessing**: Human drawing is converted to MNIST format (28×28, centered, blurred)
4. **Judging**: Discriminator scores both images (0-100%)
5. **Result**: Higher score wins; ties within ±1%

### MNIST Preprocessing

Human drawings are preprocessed to match MNIST format:
- Bounding box detection
- Scale to fit 20×20 region
- Center in 28×28 canvas
- Apply 1px blur for soft edges
- Normalize to [0, 1]

## Training Your Own Models

The game works with any conditional GAN trained on MNIST. See `notebook_cells.py` for training code supporting:

- **BCE**: Binary Cross-Entropy (original GAN)
- **LSGAN**: Least Squares GAN
- **Hinge**: Hinge loss (SAGAN/BigGAN style)
- **WGAN-GP**: Wasserstein GAN with Gradient Penalty

## License

Apache License 2.0

## Acknowledgments

- MNIST dataset by Yann LeCun et al.
- GAN architecture inspired by DCGAN and modern best practices
- Tug-of-war illustration concept for the adversarial game visualization
