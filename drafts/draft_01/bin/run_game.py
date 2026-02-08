#!/usr/bin/env python3
"""
Human vs GAN - Standalone Game Server

Usage:
    python run_game.py
    python run_game.py --port 8993 --strategy bce
    python run_game.py --classifier ./classifier/mnist_cnn_best.ckpt

Then open http://localhost:8993 in your browser.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


# ============================================
# CLASSIFIER MODEL (for judging)
# ============================================

class MNISTCNNClassifier(nn.Module):
    """
    Standalone CNN classifier matching MNISTCNN architecture.
    Used as the Judge in the game.
    """
    
    def __init__(
        self,
        width: int = 128,
        depth: int = 3,
        activation: str = "gelu",
        use_bn: bool = False,
        dropout_p: float = 0.0,
        num_classes: int = 10,
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        
        # Activation factory
        act_table = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leakyrelu": nn.LeakyReLU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
        }
        act_cls = act_table.get(activation.lower(), nn.GELU)
        
        # Build encoder
        channels = [width, 2 * width, 2 * width][:depth]
        in_ch = 1
        blocks = []
        for out_ch in channels:
            blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn))
            if use_bn:
                blocks.append(nn.BatchNorm2d(out_ch))
            blocks.append(act_cls())
            blocks.append(nn.MaxPool2d(kernel_size=2))
            in_ch = out_ch
        self.encoder = nn.Sequential(*blocks)
        
        # Classifier head
        spatial = 28 // (2 ** depth)
        feat_dim = channels[-1] * spatial * spatial
        head_layers = [
            nn.Flatten(),
            nn.Linear(feat_dim, width),
            act_cls(),
        ]
        if dropout_p > 0.0:
            head_layers.append(nn.Dropout(dropout_p))
        head_layers.append(nn.Linear(width, num_classes))
        self.head = nn.Sequential(*head_layers)
    
    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)


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
classifier = None
device = None
latent_dim = 100

# MNIST dataset for "MNIST Digit" mode
mnist_dataset = None
digit_indices = None

# Normalization constants
# GAN uses: (x - 0.5) / 0.5 -> range [-1, 1]
# Classifier uses MNIST stats: (x - 0.1307) / 0.3081
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


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
    global g_model, device, latent_dim, mnist_dataset, digit_indices
    
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


def renormalize_for_classifier(tensor_gan):
    """
    Convert tensor from GAN normalization [-1, 1] to classifier normalization.
    GAN: (x - 0.5) / 0.5 -> [-1, 1]
    Classifier: (x - MNIST_MEAN) / MNIST_STD
    
    First convert back to [0, 1], then apply classifier normalization.
    """
    # GAN [-1, 1] -> [0, 1]
    tensor_01 = (tensor_gan + 1) / 2
    # [0, 1] -> classifier normalization
    tensor_clf = (tensor_01 - MNIST_MEAN) / MNIST_STD
    return tensor_clf


@sio.event
async def judge_drawing(sid, data):
    """Score player's drawing using the classifier."""
    global classifier, device
    
    if classifier is None:
        print("[game] Error: Classifier not loaded!")
        return
    
    if sid not in _player_state:
        print("[game] Error: No round started for this player!")
        return
    
    try:
        # Get stored generator output
        state = _player_state[sid]
        gen_tensor = state["gen_tensor"]  # In GAN normalization [-1, 1]
        digit = state["digit"]
        
        # Parse player's drawing (binary Float32Array, values in [0, 1])
        image_bytes = data["image"]
        if isinstance(image_bytes, bytes):
            human_image = np.frombuffer(image_bytes, dtype=np.float32).reshape(1, 1, 28, 28)
        else:
            human_image = np.array(image_bytes, dtype=np.float32).reshape(1, 1, 28, 28)
        
        # Convert to tensor in GAN normalization [-1, 1]
        human_tensor = torch.from_numpy(human_image.copy()).to(device)
        human_tensor = human_tensor * 2 - 1
        
        # Renormalize both for classifier
        human_clf = renormalize_for_classifier(human_tensor)
        gen_clf = renormalize_for_classifier(gen_tensor)
        
        # Debug: check tensor stats
        print(f"[game] Human (clf norm) - min: {human_clf.min():.3f}, max: {human_clf.max():.3f}, mean: {human_clf.mean():.3f}")
        print(f"[game] Gen (clf norm)   - min: {gen_clf.min():.3f}, max: {gen_clf.max():.3f}, mean: {gen_clf.mean():.3f}")
        
        # Score both images using classifier
        with torch.no_grad():
            human_logits = classifier(human_clf)
            gen_logits = classifier(gen_clf)
            
            # Get probability for target digit
            human_probs = torch.softmax(human_logits, dim=1)
            gen_probs = torch.softmax(gen_logits, dim=1)
            
            human_score = human_probs[0, digit].item()
            gen_score = gen_probs[0, digit].item()
            
            # Also get predicted class
            human_pred = human_logits.argmax(dim=1).item()
            gen_pred = gen_logits.argmax(dim=1).item()
        
        print(f"[game] Classifier scores - Human: {human_score:.6f} (pred={human_pred}), GAN: {gen_score:.6f} (pred={gen_pred})")
        
        await sio.emit("game_result", {
            "human_score": float(human_score),
            "gen_score": float(gen_score),
        }, to=sid)
        
        print(f"[game] Digit {digit} ({state.get('source', 'Generator')}): Human={human_score:.1%}, GAN={gen_score:.1%}")
        
    except Exception as e:
        print(f"[game] Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================
# MAIN
# ============================================

def main():
    global g_model, classifier, device, latent_dim, mnist_dataset, digit_indices
    
    parser = argparse.ArgumentParser(description="Human vs GAN Game Server")
    parser.add_argument("--port", type=int, default=8993, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--model-dir", default="./model", help="Directory containing model files")
    parser.add_argument("--mnist-dir", default="../../dataset", help="Directory containing MNIST dataset")
    parser.add_argument("--classifier", default="../../drafts/draft_01/classifier/mnist_cnn_calibrated_best.ckpt", 
                        help="Path to classifier checkpoint")
    parser.add_argument("--strategy", default="wgan-gp", choices=["bce", "lsgan", "hinge", "wgan-gp"],
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
    
    # Load classifier
    classifier_path = Path(args.classifier)
    if classifier_path.exists():
        try:
            # Load Lightning checkpoint
            checkpoint = torch.load(classifier_path, map_location=device, weights_only=False)
            
            # Extract hyperparameters if available
            hparams = checkpoint.get("hyper_parameters", {})
            width = hparams.get("width", 128)
            depth = hparams.get("depth", 3)
            activation = hparams.get("activation", "relu")
            use_bn = hparams.get("use_bn", False)
            dropout_p = hparams.get("dropout_p", 0.1)
            
            print(f"[game] Classifier hparams: width={width}, depth={depth}, activation={activation}, use_bn={use_bn}, dropout_p={dropout_p}")
            
            # Create model and load weights
            classifier = MNISTCNNClassifier(
                width=width,
                depth=depth,
                activation=activation,
                use_bn=use_bn,
                dropout_p=dropout_p,
            ).to(device)
            
            # Lightning saves state_dict under "state_dict" key
            state_dict = checkpoint.get("state_dict", checkpoint)
            
            # Remove "model." prefix if present (Lightning adds module prefixes)
            new_state_dict = {}
            for k, v in state_dict.items():
                # Remove common prefixes
                new_key = k.replace("model.", "").replace("encoder.", "encoder.").replace("head.", "head.")
                new_state_dict[new_key] = v
            
            classifier.load_state_dict(new_state_dict, strict=False)
            classifier.eval()
            print(f"[game] Loaded classifier: {classifier_path}")
        except Exception as e:
            print(f"[game] Warning: Could not load classifier: {e}")
            import traceback
            traceback.print_exc()
            classifier = None
    else:
        print(f"[game] Warning: Classifier not found: {classifier_path}")
        classifier = None
    
    # Load GAN models
    model_dir = Path(args.model_dir)
    g_path = model_dir / f"G_{args.strategy}.pt"
    
    if not g_path.exists():
        print(f"[game] Error: Generator model not found!")
        print(f"       Expected: {g_path}")
        print(f"\n       Available files in {model_dir}:")
        if model_dir.exists():
            for f in model_dir.iterdir():
                print(f"         - {f.name}")
        return
    
    g_model = Generator(latent_dim=latent_dim).to(device)
    g_model.load_state_dict(torch.load(g_path, map_location=device, weights_only=True))
    g_model.eval()
    print(f"[game] Loaded generator: {g_path}")
    
    # Check if classifier is available
    if classifier is None:
        print(f"\n[game] WARNING: No classifier loaded! Game will not work properly.")
        print(f"       Please provide a valid classifier checkpoint with --classifier")
    
    # Run server
    print(f"\n[game] Starting server at http://localhost:{args.port}")
    print(f"[game] Strategy: {args.strategy.upper()}")
    print(f"[game] Judge: {'Classifier' if classifier else 'NONE'}")
    print(f"[game] Press Ctrl+C to stop\n")
    
    uvicorn.run(asgi_app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
