"""
GAN Game Server — Socket.IO server for the "Beat the GAN" game.

Usage from notebook:
    from bin.gan_game import start_game_server, set_models
    
    # Load trained models
    g_model = Generator().to(device)
    g_model.load_state_dict(torch.load('model/G_bce.pt'))
    g_model.eval()
    
    d_model = Discriminator(use_sigmoid=True).to(device)
    d_model.load_state_dict(torch.load('model/D_bce.pt'))
    d_model.eval()
    
    # Start game server
    set_models(g_model, d_model, device)
    start_game_server(port=8993)
"""

import asyncio
import threading

import numpy as np
import torch
import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# --- Async Socket.IO server ---
_sio_async = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

# --- FastAPI + ASGI wrapping ---
_fast = FastAPI()
_app = socketio.ASGIApp(_sio_async, _fast)

# Event loop reference
_loop = None

# Model references (set by notebook)
_g_model = None
_d_model = None
_device = None
_latent_dim = 100


def _load_html():
    """Load the game HTML from web/ sibling folder."""
    import pathlib
    html_path = pathlib.Path(__file__).parent.parent / "web" / "game.html"
    return html_path.read_text()


@_fast.get("/")
def index():
    return HTMLResponse(_load_html())


@_sio_async.event
async def connect(sid, environ):
    print(f"[gan_game] Player connected: {sid}")


@_sio_async.event
async def disconnect(sid):
    print(f"[gan_game] Player disconnected: {sid}")


@_sio_async.event
async def judge_drawing(sid, data):
    """
    Handle a player's drawing submission.
    
    Args:
        data: {
            "image": list of 784 floats (28x28 grayscale, 0-1),
            "digit": int (0-9)
        }
    
    Emits:
        "game_result": {
            "gen_image": bytes (784 uint8 values),
            "human_score": float (0-1),
            "gen_score": float (0-1)
        }
    """
    if _g_model is None or _d_model is None:
        print("[gan_game] Error: Models not loaded!")
        return
    
    try:
        # Parse player's drawing
        human_image = np.array(data["image"], dtype=np.float32).reshape(1, 1, 28, 28)
        digit = int(data["digit"])
        
        # Convert to tensor, normalize to [-1, 1]
        human_tensor = torch.from_numpy(human_image).to(_device)
        human_tensor = human_tensor * 2 - 1  # [0,1] -> [-1,1]
        
        # Create label tensor
        label = torch.tensor([[digit]], device=_device)
        
        # Generate image from Generator
        with torch.no_grad():
            z = torch.randn(1, _latent_dim, device=_device)
            gen_tensor = _g_model(z, label)
        
        # Score both images with Discriminator
        with torch.no_grad():
            human_score = _d_model(human_tensor, label)
            gen_score = _d_model(gen_tensor, label)
            
            # Apply sigmoid if not already applied (BCE models have sigmoid in forward)
            # If scores are outside [0,1], apply sigmoid
            if human_score.min() < 0 or human_score.max() > 1:
                human_score = torch.sigmoid(human_score)
            if gen_score.min() < 0 or gen_score.max() > 1:
                gen_score = torch.sigmoid(gen_score)
        
        # Convert generator output to bytes for transmission
        gen_image_np = ((gen_tensor[0, 0].cpu().float() + 1) / 2 * 255).clamp(0, 255).byte().numpy()
        
        # Send result back to player
        await _sio_async.emit("game_result", {
            "gen_image": gen_image_np.tobytes(),
            "human_score": float(human_score.item()),
            "gen_score": float(gen_score.item()),
        }, to=sid)
        
        print(f"[gan_game] Digit {digit}: Human={human_score.item():.3f}, Gen={gen_score.item():.3f}")
        
    except Exception as e:
        print(f"[gan_game] Error processing drawing: {e}")


# --- Server lifecycle ---
_server_thread = None


def set_models(g_model, d_model, device, latent_dim=100):
    """Set the Generator and Discriminator models for the game."""
    global _g_model, _d_model, _device, _latent_dim
    _g_model = g_model
    _d_model = d_model
    _device = device
    _latent_dim = latent_dim
    print("[gan_game] Models loaded successfully")


def start_game_server(host="0.0.0.0", port=8993):
    """Start the game server in a background daemon thread."""
    global _server_thread, _loop
    
    if _g_model is None or _d_model is None:
        print("[gan_game] Warning: Models not set! Call set_models() first.")
    
    if _server_thread and _server_thread.is_alive():
        print(f"[gan_game] Already running on port {port}")
        return

    def _run():
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        config = uvicorn.Config(_app, host=host, port=port, log_level="warning", loop="asyncio")
        server = uvicorn.Server(config)
        _loop.run_until_complete(server.serve())

    _server_thread = threading.Thread(target=_run, daemon=True)
    _server_thread.start()

    # Wait for the event loop to be ready
    import time
    for _ in range(50):
        if _loop is not None:
            break
        time.sleep(0.05)

    print(f"[gan_game] Game server at http://localhost:{port}")
