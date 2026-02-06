"""
GAN Live Monitor — Socket.IO server for real-time training visualization.

Usage from notebook:
    Cell (globals):   LIVE_MONITOR = True
    Cell (server):    from bin.gan_monitor import start_server, emit_frames, emit_done
                      if LIVE_MONITOR: start_server(port=8765)
    Cell (training):  if LIVE_MONITOR: emit_frames(images, labels, step, g_loss, d_loss)
    Cell (after):     if LIVE_MONITOR: emit_done()
"""

import asyncio
import threading

import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# --- Async Socket.IO server (runs inside uvicorn's event loop) ---
_sio_async = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

# --- FastAPI + ASGI wrapping ---
_fast = FastAPI()
_app = socketio.ASGIApp(_sio_async, _fast)

# Event loop reference (set when server starts)
_loop = None


def _load_html():
    """Load the HTML client from web/ sibling folder."""
    import pathlib
    html_path = pathlib.Path(__file__).parent.parent / "web" / "index.html"
    return html_path.read_text()


@_fast.get("/")
def index():
    return HTMLResponse(_load_html())


@_sio_async.event
async def connect(sid, environ):
    print(f"[gan_monitor] Client connected: {sid}")


@_sio_async.event
async def disconnect(sid):
    print(f"[gan_monitor] Client disconnected: {sid}")


# --- Server lifecycle ---
_server_thread = None


def start_server(host="0.0.0.0", port=8765):
    """Start the monitor server in a background daemon thread."""
    global _server_thread, _loop
    if _server_thread and _server_thread.is_alive():
        print(f"[gan_monitor] Already running on port {port}")
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

    print(f"[gan_monitor] Live monitor at http://localhost:{port}")


# --- Emit helpers ---
def emit_frames(images, labels, step, g_loss, d_loss):
    """
    Send generated images to all connected browsers.

    Args:
        images: Tensor of shape (N, 1, 28, 28) in [-1, 1]
        labels: Tensor of shape (N, 1) with class indices
        step:   Current training step (int)
        g_loss: Generator loss (float)
        d_loss: Discriminator loss (float)
    """
    frames = []
    for i in range(images.shape[0]):
        # [-1, 1] -> [0, 255] as raw bytes (784 bytes for 28x28)
        img_np = ((images[i, 0].detach().cpu().float() + 1) / 2 * 255).clamp(0, 255).byte().numpy()
        frames.append({
            "index": i,
            "label": int(labels[i].item()),
            "image": img_np.tobytes(),
        })

    payload = {
        "step": step,
        "g_loss": round(g_loss, 4),
        "d_loss": round(d_loss, 4),
        "frames": frames,
    }

    # Bridge sync caller -> async Socket.IO server
    if _loop is not None and _loop.is_running():
        future = asyncio.run_coroutine_threadsafe(_sio_async.emit("batch", payload), _loop)
        future.result(timeout=2)


def emit_done():
    """Signal that training has completed. Stops the timer in connected browsers."""
    if _loop is not None and _loop.is_running():
        future = asyncio.run_coroutine_threadsafe(_sio_async.emit("done", {}), _loop)
        future.result(timeout=2)
    print("[gan_monitor] Training complete signal sent")
