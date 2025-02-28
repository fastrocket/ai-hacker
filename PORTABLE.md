
# Cross-Platform FastAPI Deployment Guide

This document outlines key considerations and best practices for ensuring your FastAPI application works consistently across Replit, Windows, and macOS environments.

## IP Binding and Port Configuration

### Best Practices

- Always bind to `0.0.0.0` instead of `localhost` or `127.0.0.1` to make your application accessible from outside the container/VM:

```python
uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

- Use environment variables for port configuration to support different deployment environments:

```python
import os
port = int(os.environ.get("PORT", 8000))
uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
```

## Path Handling

### Windows vs. Unix Path Differences

Windows uses backslashes (`\`) while Unix-based systems (macOS, Linux, Replit) use forward slashes (`/`) for file paths. To ensure cross-platform compatibility:

1. Always use `os.path.join()` for path construction:

```python
static_path = os.path.join(os.path.dirname(__file__), "static")
```

2. Normalize paths when working with URLs:

```python
normalized_path = path.lstrip('/').replace('/', os.sep)
file_path = os.path.join(static_dir, normalized_path)
```

3. Use `Path` from `pathlib` for modern path handling:

```python
from pathlib import Path
static_path = Path(__file__).parent / "static"
```

## Static File Serving

### Consistent Static File Configuration

1. For FastAPI, mount static files using absolute paths:

```python
from pathlib import Path
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
```

2. For simple HTTP servers, ensure proper path resolution:

```python
# Define absolute path for static directory
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# Handle static file requests
if path.startswith('/static/'):
    path = path[7:]  # Remove '/static/' prefix
    normalized_path = path.replace('/', os.sep)
    file_path = os.path.join(STATIC_DIR, normalized_path)
```

## WebSocket Configuration

### Cross-Platform WebSocket Setup

1. Use `0.0.0.0` for binding WebSocket servers:

```python
server = await websockets.serve(handler, "0.0.0.0", websocket_port)
```

2. For client connections, adapt the WebSocket URL based on the environment:

```javascript
// In JavaScript client code
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${wsProtocol}//${window.location.host}/ws/generate`;
const socket = new WebSocket(wsUrl);
```

## CORS Configuration

### Universal CORS Setup

Configure CORS to work in both development and production:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## External Service Communication

### Communicating with Local Services (e.g., Ollama)

1. Use environment variables to configure service URLs:

```python
import os
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api")
```

2. Implement fallbacks when services are unavailable:

```python
try:
    # Try to connect to Ollama API
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{OLLAMA_API_URL}/tags", timeout=2) as response:
            if response.status == 200:
                return {"status": "running"}
except Exception as e:
    logger.warning(f"Ollama API connection failed: {e}")
    return {"status": "not_running", "message": str(e)}
```

## Deployment in Replit

For deploying the application on Replit, set the proper configuration in the "Deploy" tab:

1. Set the run command to:
```
uvicorn main:app --host 0.0.0.0 --port 8080
```

2. Ensure all dependencies are listed in `requirements.txt`.

3. For WebSocket support, ensure your deployment uses a compatible service tier.

## Testing

Before deploying, test your application in various environments:

1. Test locally with:
```
python main.py
```

2. Test by explicitly specifying host and port:
```
uvicorn main:app --host 0.0.0.0 --port 8000
```

3. Test WebSocket functionality separately to ensure cross-platform compatibility.

By following these guidelines, your FastAPI application should work consistently across Replit, Windows, and macOS environments.
