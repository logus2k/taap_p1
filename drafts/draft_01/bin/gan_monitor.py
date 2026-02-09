"""
GAN Live Monitor — Socket.IO server for real-time training visualization.

Usage from notebook:
    Cell (globals):   LIVE_MONITOR = True
    Cell (server):    from bin.gan_monitor import start_server, emit_frames, emit_done, emit_benchmark_start, emit_strategy_start, emit_strategy_end
                      if LIVE_MONITOR: start_server(port=8765)
    
    Notebook display (optional):
        from bin.gan_monitor import setup_notebook_display
        setup_notebook_display(rows=4, cols=4)  # Creates live-updating grid in notebook
    
    Benchmark flow:
        emit_benchmark_start(strategies, num_steps)  # at benchmark start
        for strategy in strategies:
            emit_strategy_start(strategy_name, strategy_index, total_strategies)
            for step in training:
                emit_frames(images, labels, step, g_loss, d_loss)
            emit_strategy_end(strategy_name, fid, kid_mean, kid_std, training_time)
        emit_done()  # all complete
"""

import asyncio
import pathlib
import threading

import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Get the web directory path (try both locations)
_web_dir = pathlib.Path(__file__).parent.parent / "web"
if not _web_dir.exists():
    _web_dir = pathlib.Path(__file__).parent / "web"

# --- FastAPI app ---
app = FastAPI()

# --- Async Socket.IO server ---
# Support both direct access (/socket.io) and nginx proxy (/gan/socket.io)
# The client (live_monitor.js) detects which path to use based on window.location.pathname
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

# Mount static files for CSS, JS, images, etc.
for subdir in ["libraries", "styles", "scripts", "images", "css", "js"]:
    subdir_path = _web_dir / subdir
    if subdir_path.exists():
        app.mount(f"/{subdir}", StaticFiles(directory=subdir_path), name=subdir)

# Create ASGI app combining Socket.IO and FastAPI
# socketio_path handles both /socket.io and /gan/socket.io via nginx rewrite
asgi_app = socketio.ASGIApp(sio, app)

# Event loop reference (set when server starts)
_loop = None

# --- Notebook display subscriber ---
_notebook_display = None  # Will hold {'fig': figure, 'axes': axes, 'enabled': bool}


def _load_html():
    """Load the HTML client from web/ sibling folder."""
    html_path = _web_dir / "index.html"
    return html_path.read_text()


@app.get("/")
def index():
    return HTMLResponse(_load_html())


@sio.event
async def connect(sid, environ):
    print(f"[gan_monitor] Client connected: {sid}")


@sio.event
async def disconnect(sid):
    print(f"[gan_monitor] Client disconnected: {sid}")


# --- Notebook display subscriber ---
def setup_notebook_display(rows=4, cols=4, figsize=None):
    """
    Set up a live-updating matplotlib display in the notebook.
    
    Args:
        rows: Number of rows in the image grid (default: 4)
        cols: Number of columns in the image grid (default: 4)
        figsize: Figure size tuple, or None for auto (default: None)
    
    Returns:
        (fig, axes) tuple for reference
    
    Usage:
        fig, axes = setup_notebook_display(rows=4, cols=4)
        # Now emit_frames() will automatically update this display
    """
    global _notebook_display
    
    import matplotlib.pyplot as plt
    from IPython.display import display
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (cols * 1.5, rows * 1.5)
    
    # Create figure with dark background to match GAN output style
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.patch.set_facecolor('#1a1a2e')
    
    # Flatten axes for easy iteration
    axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]
    
    # Initialize empty cells
    for ax in axes_flat:
        ax.set_facecolor('#1a1a2e')
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    # Store reference for emit_frames to use
    _notebook_display = {
        'fig': fig,
        'axes': axes_flat,
        'enabled': True,
        'rows': rows,
        'cols': cols,
    }
    
    # Display the figure (will be updated in-place)
    display(fig)
    
    print(f"[gan_monitor] Notebook display ready ({rows}x{cols} grid)")
    return fig, axes


def disable_notebook_display():
    """Disable notebook display updates."""
    global _notebook_display
    if _notebook_display:
        _notebook_display['enabled'] = False
        print("[gan_monitor] Notebook display disabled")


def enable_notebook_display():
    """Re-enable notebook display updates."""
    global _notebook_display
    if _notebook_display:
        _notebook_display['enabled'] = True
        print("[gan_monitor] Notebook display enabled")


def _update_notebook_display(images, labels):
    """Update the notebook matplotlib display with new images."""
    global _notebook_display
    
    if _notebook_display is None or not _notebook_display.get('enabled', False):
        return
    
    try:
        fig = _notebook_display['fig']
        axes = _notebook_display['axes']
        
        n_images = min(len(images), len(list(axes)))
        
        for i in range(n_images):
            ax = list(axes)[i]
            ax.clear()
            ax.set_facecolor('#1a1a2e')
            
            # Convert tensor to numpy: [-1, 1] -> [0, 1]
            img = (images[i, 0].detach().cpu().float() + 1) / 2
            img = img.clamp(0, 1).numpy()
            
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            
            # Add label as title
            if labels is not None and i < len(labels):
                label = int(labels[i].item()) if hasattr(labels[i], 'item') else int(labels[i])
                ax.set_title(str(label), color='white', fontsize=10, pad=2)
        
        # Refresh the display
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        
    except Exception as e:
        # Silently handle errors to not interrupt training
        pass


# --- Server lifecycle ---
_server_thread = None


def start_server(host="0.0.0.0", port=8765, open_browser=True):
    """Start the monitor server in a background daemon thread.
    
    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to listen on (default: 8765)
        open_browser: Automatically open the monitor in a web browser (default: True)
    """
    global _server_thread, _loop
    
    print(f"[gan_monitor] Web directory: {_web_dir} (exists: {_web_dir.exists()})")
    
    # List mounted static directories
    for subdir in ["libraries", "styles", "scripts", "images"]:
        subdir_path = _web_dir / subdir
        if subdir_path.exists():
            print(f"[gan_monitor] Static mount: /{subdir} -> {subdir_path}")
    
    if _server_thread and _server_thread.is_alive():
        print(f"[gan_monitor] Already running on port {port}")
        return

    def _run():
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        config = uvicorn.Config(asgi_app, host=host, port=port, log_level="warning", loop="asyncio")
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

    url = f"http://localhost:{port}"
    print(f"[gan_monitor] Live monitor at {url}")
    
    # Open browser if requested
    if open_browser:
        import webbrowser
        webbrowser.open(url)


def _emit(event, payload):
    """Helper to emit events from sync code."""
    if _loop is not None and _loop.is_running():
        future = asyncio.run_coroutine_threadsafe(sio.emit(event, payload), _loop)
        future.result(timeout=2)


# --- Benchmark lifecycle events ---
def emit_benchmark_start(strategies, num_steps):
    """Signal the start of a benchmark run."""
    _emit("benchmark_start", {
        "strategies": strategies,
        "num_steps": num_steps,
        "total_strategies": len(strategies),
    })
    print(f"[gan_monitor] Benchmark started: {strategies}")


def emit_strategy_start(strategy_name, strategy_index, total_strategies):
    """Signal the start of training for a specific strategy."""
    _emit("strategy_start", {
        "strategy": strategy_name,
        "index": strategy_index,
        "total": total_strategies,
    })
    print(f"[gan_monitor] Strategy started: {strategy_name} ({strategy_index + 1}/{total_strategies})")


def emit_strategy_end(strategy_name, fid, kid_mean, kid_std, training_time):
    """Signal the end of training for a specific strategy with results."""
    _emit("strategy_end", {
        "strategy": strategy_name,
        "fid": round(fid, 2),
        "kid_mean": round(kid_mean, 4),
        "kid_std": round(kid_std, 4),
        "training_time": round(training_time, 1),
    })
    print(f"[gan_monitor] Strategy complete: {strategy_name} (FID: {fid:.2f})")


# --- Frame emission ---
def emit_frames(images, labels, step, g_loss, d_loss, num_steps=None):
    """
    Send generated images to all connected browsers and notebook display.

    Args:
        images: Tensor of shape (N, 1, 28, 28) in [-1, 1]
        labels: Tensor of shape (N, 1) with class indices
        step:   Current training step (int)
        g_loss: Generator loss (float)
        d_loss: Discriminator loss (float)
        num_steps: Total steps (optional, for progress calculation)
    """
    # Update notebook display (subscriber 1)
    _update_notebook_display(images, labels)
    
    # Send to websocket clients (subscriber 2)
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
    
    if num_steps is not None:
        payload["num_steps"] = num_steps
        payload["progress"] = round(step / num_steps * 100, 1)

    _emit("batch", payload)


def emit_done():
    """Signal that all training has completed."""
    _emit("done", {})
    print("[gan_monitor] Benchmark complete signal sent")
