#!/usr/bin/env python3
"""
Human vs GAN - Standalone Game Server

Usage:
    python run_game.py
    python run_game.py --port 8993 --strategy bce
    python run_game.py --model-dir ./model --strategy lsgan

Then open http://localhost:8993 in your browser.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torchvision import datasets, transforms
import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# ============================================
# MODEL DEFINITIONS (must match training)
# ============================================

class Generator(nn.Module):
    """Generator with PixelShuffle upsampling and ICNR initialization."""
    
    def __init__(self, latent_dim=100, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.ReLU(),
        )
        
        # PixelShuffle upsampling blocks
        self.conv1 = nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1)
        self.ps1 = nn.PixelShuffle(2)
        self.bn1 = nn.BatchNorm2d(128, momentum=0.8)
        
        self.conv2 = nn.Conv2d(128, 64 * 4, kernel_size=3, padding=1)
        self.ps2 = nn.PixelShuffle(2)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.8)
        
        self.output_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)
    
    def forward(self, z, labels):
        label_embed = self.label_embedding(labels).squeeze(1)
        x = self.fc(z * label_embed)
        x = x.view(-1, 128, 7, 7)
        
        x = torch.relu(self.bn1(self.ps1(self.conv1(x))))
        x = torch.relu(self.bn2(self.ps2(self.conv2(x))))
        x = torch.tanh(self.output_conv(x))
        return x


class Discriminator(nn.Module):
    """Discriminator with spectral normalization."""
    
    def __init__(self, num_classes=10, use_sigmoid=True):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.label_embedding = nn.Embedding(num_classes, 28 * 28)
        
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Flatten(),
            spectral_norm(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            nn.Dropout(0.25),
            spectral_norm(nn.Linear(512, 1)),
        )
    
    def forward(self, img, labels):
        batch_size = img.size(0)
        label_embed = self.label_embedding(labels)
        label_embed = label_embed.view(batch_size, 1, 28, 28)
        x = torch.cat([img, label_embed], dim=1)
        out = self.model(x)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out


# ============================================
# SOCKET.IO SERVER
# ============================================

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = FastAPI()

# Mount static files (fonts, images, scripts, etc.)
def get_web_dir():
    """Find the web/ directory."""
    web_path = Path(__file__).parent / "web"
    if not web_path.exists():
        web_path = Path(__file__).parent.parent / "web"
    return web_path

web_dir = get_web_dir()
if web_dir.exists():
    # Mount subdirectories if they exist
    for subdir in ["fonts", "images", "scripts", "styles", "libraries"]:
        subdir_path = web_dir / subdir
        if subdir_path.exists():
            app.mount(f"/{subdir}", StaticFiles(directory=subdir_path), name=subdir)
    # Also serve web/ root for any loose files
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

asgi_app = socketio.ASGIApp(sio, app)

# Global model references
g_model = None
d_model = None
device = None
latent_dim = 100

# MNIST dataset for "MNIST Digit" mode
mnist_dataset = None
digit_indices = None  # {0: [idx, idx, ...], 1: [...], ...}


def load_html():
    """Load game.html from web/ folder."""
    html_path = Path(__file__).parent / "web" / "game.html"
    if not html_path.exists():
        # Try sibling directory
        html_path = Path(__file__).parent.parent / "web" / "game.html"
    return html_path.read_text()


@app.get("/")
def index():
    return HTMLResponse(load_html())


@sio.event
async def connect(sid, environ):
    print(f"[game] Player connected: {sid}")


@sio.event
async def disconnect(sid):
    print(f"[game] Player disconnected: {sid}")
    # Clean up stored state
    if sid in _player_state:
        del _player_state[sid]


# Store generator output per player session
_player_state = {}


@sio.event
async def start_round(sid, data):
    """Generate digit for new round."""
    global g_model, d_model, device, latent_dim, mnist_dataset, digit_indices
    
    print(f"[game] start_round received data: {data}")
    
    digit = int(data["digit"])
    use_mnist = data.get("use_mnist", False)
    
    print(f"[game] use_mnist={use_mnist}, mnist_dataset={mnist_dataset is not None}, digit_indices={digit_indices is not None}")
    
    try:
        label = torch.tensor([[digit]], device=device)  # Shape: (1, 1) as used in training
        
        if use_mnist and mnist_dataset is not None and digit_indices is not None:
            # Use real MNIST digit
            idx = random.choice(digit_indices[digit])
            img, _ = mnist_dataset[idx]
            # img is already a tensor in [-1, 1] from transform
            gen_tensor = img.unsqueeze(0).to(device)  # Shape: (1, 1, 28, 28)
            source = "MNIST"
        else:
            # Use Generator
            if g_model is None:
                print("[game] Error: Generator not loaded!")
                return
            
            with torch.no_grad():
                z = torch.randn(1, latent_dim, device=device)
                gen_tensor = g_model(z, label)
            source = "Generator"
        
        # Store for later scoring
        _player_state[sid] = {
            "digit": digit,
            "gen_tensor": gen_tensor,
            "label": label,
            "source": source,
        }
        
        # Convert to binary for client
        gen_image_np = ((gen_tensor[0, 0].cpu().float() + 1) / 2 * 255).clamp(0, 255).byte().numpy()
        
        await sio.emit("round_ready", {
            "gen_image": bytes(gen_image_np.flatten()),
        }, to=sid)
        
        print(f"[game] Round started: digit {digit} ({source})")
        
    except Exception as e:
        print(f"[game] Error generating: {e}")
        import traceback
        traceback.print_exc()


@sio.event
async def judge_drawing(sid, data):
    """Score player's drawing against stored generator output."""
    global d_model, device
    
    if d_model is None:
        print("[game] Error: Discriminator not loaded!")
        return
    
    if sid not in _player_state:
        print("[game] Error: No round started for this player!")
        return
    
    try:
        # Get stored generator output
        state = _player_state[sid]
        gen_tensor = state["gen_tensor"]
        label = state["label"]
        
        # Parse player's drawing (binary Float32Array)
        image_bytes = data["image"]
        if isinstance(image_bytes, bytes):
            human_image = np.frombuffer(image_bytes, dtype=np.float32).reshape(1, 1, 28, 28)
        else:
            human_image = np.array(image_bytes, dtype=np.float32).reshape(1, 1, 28, 28)
        
        # Convert to tensor, normalize to [-1, 1]
        human_tensor = torch.from_numpy(human_image.copy()).to(device)
        human_tensor = human_tensor * 2 - 1
        
        # Debug: check tensor stats
        print(f"[game] Human tensor - min: {human_tensor.min():.3f}, max: {human_tensor.max():.3f}, mean: {human_tensor.mean():.3f}")
        print(f"[game] Gen tensor   - min: {gen_tensor.min():.3f}, max: {gen_tensor.max():.3f}, mean: {gen_tensor.mean():.3f}")
        
        # Score both images
        with torch.no_grad():
            human_score_raw = d_model(human_tensor, label)
            gen_score_raw = d_model(gen_tensor, label)
            
            print(f"[game] Raw scores - Human: {human_score_raw.item():.4f}, GAN: {gen_score_raw.item():.4f}")
            
            # Apply sigmoid if needed (for non-BCE strategies where model doesn't apply sigmoid)
            if human_score_raw.min() < 0 or human_score_raw.max() > 1:
                human_score = torch.sigmoid(human_score_raw)
                gen_score = torch.sigmoid(gen_score_raw)
                print(f"[game] After sigmoid - Human: {human_score.item():.4f}, GAN: {gen_score.item():.4f}")
            else:
                human_score = human_score_raw
                gen_score = gen_score_raw
        
        await sio.emit("game_result", {
            "human_score": float(human_score.item()),
            "gen_score": float(gen_score.item()),
        }, to=sid)
        
        print(f"[game] Digit {state['digit']} ({state.get('source', 'Generator')}): Human={human_score.item():.1%}, GAN={gen_score.item():.1%}")
        
    except Exception as e:
        print(f"[game] Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================
# MAIN
# ============================================

def main():
    global g_model, d_model, device, latent_dim, mnist_dataset, digit_indices
    
    parser = argparse.ArgumentParser(description="Human vs GAN Game Server")
    parser.add_argument("--port", type=int, default=8993, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--model-dir", default="./model", help="Directory containing model files")
    parser.add_argument("--mnist-dir", default="../../dataset", help="Directory containing MNIST dataset")
    parser.add_argument("--strategy", default="bce", choices=["bce", "lsgan", "hinge", "wgan-gp"],
                        help="Which trained strategy to use")
    parser.add_argument("--latent-dim", type=int, default=100, help="Latent dimension")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    args = parser.parse_args()
    
    # Setup device
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[game] Using device: {device}")
    
    latent_dim = args.latent_dim
    
    # Load MNIST dataset for "MNIST Digit" mode
    mnist_path = Path(args.mnist_dir)
    if mnist_path.exists():
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # Scale to [-1, 1]
            ])
            mnist_dataset = datasets.MNIST(root=args.mnist_dir, train=True, download=False, transform=transform)
            
            # Build index: one pass through all labels
            digit_indices = {d: [] for d in range(10)}
            for i in range(len(mnist_dataset)):
                _, label = mnist_dataset[i]
                digit_indices[label].append(i)
            
            total = sum(len(v) for v in digit_indices.values())
            print(f"[game] Loaded MNIST dataset: {total} samples indexed")
        except Exception as e:
            print(f"[game] Warning: Could not load MNIST dataset: {e}")
            mnist_dataset = None
            digit_indices = None
    else:
        print(f"[game] Warning: MNIST directory not found: {mnist_path}")
        mnist_dataset = None
        digit_indices = None
    
    # Load models
    model_dir = Path(args.model_dir)
    g_path = model_dir / f"G_{args.strategy}.pt"
    d_path = model_dir / f"D_{args.strategy}.pt"
    
    if not g_path.exists() or not d_path.exists():
        print(f"[game] Error: Model files not found!")
        print(f"       Expected: {g_path}")
        print(f"       Expected: {d_path}")
        print(f"\n       Available files in {model_dir}:")
        if model_dir.exists():
            for f in model_dir.iterdir():
                print(f"         - {f.name}")
        return
    
    # Use sigmoid for BCE, not for others
    use_sigmoid = (args.strategy == "bce")
    
    g_model = Generator(latent_dim=latent_dim).to(device)
    g_model.load_state_dict(torch.load(g_path, map_location=device, weights_only=True))
    g_model.eval()
    print(f"[game] Loaded generator: {g_path}")
    
    d_model = Discriminator(use_sigmoid=use_sigmoid).to(device)
    d_model.load_state_dict(torch.load(d_path, map_location=device, weights_only=True))
    d_model.eval()
    print(f"[game] Loaded discriminator: {d_path} (use_sigmoid={use_sigmoid})")
    
    # Run server
    print(f"\n[game] Starting server at http://localhost:{args.port}")
    print(f"[game] Strategy: {args.strategy.upper()}")
    print(f"[game] Press Ctrl+C to stop\n")
    
    uvicorn.run(asgi_app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
