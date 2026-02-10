"""
GAN Live Monitor — Multiprocessing version for real-time visualization.
Separates the server and image processing from the training process to prevent divergence.
"""

import multiprocessing as mp
import time
import numpy as np
import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pathlib

# --- Global process handle ---
_server_process = None
_tx_queue = mp.Queue(maxsize=10) # Buffer to prevent memory bloat

def _load_html():
    """Load the HTML client from web/ sibling folder."""
    try:
        html_path = pathlib.Path(__file__).parent.parent / "web" / "index.html"
        return html_path.read_text()
    except Exception:
        return "<h1>Dashboard HTML not found.</h1>"

def _server_worker(queue, host, port):
    """The entry point for the background process."""
    sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
    fast = FastAPI()
    app = socketio.ASGIApp(sio, fast)

    @fast.get("/")
    def index():
        return HTMLResponse(_load_html())

    @sio.event
    async def connect(sid, environ):
        pass

    # Background task to drain the queue and emit to web clients
    async def queue_reader():
        while True:
            try:
                # Use a loop to get data from the multiprocessing queue
                if not queue.empty():
                    event, payload = queue.get()
                    if event == "SHUTDOWN":
                        break
                    
                    # Process raw images here in the background process
                    if event == "batch_raw":
                        payload = _process_batch_raw(payload)
                        event = "batch"
                    
                    await sio.emit(event, payload)
                else:
                    import asyncio
                    await asyncio.sleep(0.01) # Don't spin the CPU
            except Exception:
                continue

    @fast.on_event("startup")
    async def startup_event():
        import asyncio
        asyncio.create_task(queue_reader())

    uvicorn.run(app, host=host, port=port, log_level="error")

def _process_batch_raw(data):
    """Converts numpy data to serializable frames (Executed in background process)."""
    images = data.pop("images") # (B, C, H, W)
    labels = data.pop("labels")
    
    frames = []
    # Assumes images are in range [-1, 1]
    for i in range(min(len(images), 8)): # Limit to 8 images to save bandwidth
        img_np = ((images[i, 0] + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        frames.append({
            "index": i,
            "label": int(labels[i]) if labels is not None else 0,
            "image": img_np.tobytes(), 
        })
    
    data["frames"] = frames
    return data

# --- Public API ---

def start_server(host="0.0.0.0", port=8765):
    global _server_process
    if _server_process and _server_process.is_alive():
        return

    _server_process = mp.Process(
        target=_server_worker, 
        args={_tx_queue, host, port}, 
        daemon=True
    )
    _server_process.start()
    print(f"[gan_monitor] Isolated monitor started at http://localhost:{port}")

def emit_frames(images, labels, step, g_loss, d_loss, num_steps=None):
    """
    Non-blocking: Tensors are moved to CPU immediately so the background 
    process can work on them without touching the GPU state.
    """
    try:
        # 1. Immediately move to CPU and convert to Numpy to 'detach' from CUDA
        # This is the most critical step for WGAN-GP stability.
        imgs_cpu = images.detach().cpu().numpy()
        lbls_cpu = labels.detach().cpu().numpy() if labels is not None else None
        
        payload = {
            "images": imgs_cpu,
            "labels": lbls_cpu,
            "step": step,
            "g_loss": float(g_loss),
            "d_loss": float(d_loss),
            "num_steps": num_steps,
            "progress": round(step / num_steps * 100, 1) if num_steps else 0
        }
        
        # 2. Put in queue. If queue is full, training doesn't wait (frame dropping).
        _tx_queue.put_nowait(("batch_raw", payload))
    except Exception:
        pass 

def emit_benchmark_start(strategies, num_steps):
    _tx_queue.put(("benchmark_start", {
        "strategies": strategies,
        "num_steps": num_steps,
        "total_strategies": len(strategies),
    }))

def emit_strategy_start(strategy_name, strategy_index, total_strategies):
    _tx_queue.put(("strategy_start", {
        "strategy": strategy_name,
        "index": strategy_index,
        "total": total_strategies,
    }))

def emit_strategy_end(strategy_name, fid, kid_mean, kid_std, training_time):
    _tx_queue.put(("strategy_end", {
        "strategy": strategy_name,
        "fid": round(fid, 2),
        "kid_mean": round(kid_mean, 4),
        "kid_std": round(kid_std, 4),
        "training_time": round(training_time, 1),
    }))

def emit_done():
    _tx_queue.put(("done", {}))
