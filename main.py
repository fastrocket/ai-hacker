from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import json
import subprocess
import asyncio
import aiohttp
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict
import logging
import requests
import signal
import threading
import time
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create an event to signal when the server should stop
stop_event = threading.Event()

def signal_handler(sig, frame):
    logger.info("Ctrl+C pressed. Shutting down server...")
    stop_event.set()

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api"

# Store active WebSocket connections
active_connections = {}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("static/index.html")

@app.get("/api/status")
async def check_ollama_status():
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags", timeout=2)
        if response.status_code == 200:
            return {"status": "running"}
        else:
            return {"status": "error"}
    except:
        return {"status": "not_running"}

@app.get("/api/models")
async def get_models():
    """Get available Ollama models."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    return {"models": models}
                else:
                    return {"models": []}
    except Exception as e:
        print(f"Error fetching models: {e}")
        return {"models": []}

@app.get("/api/ollama-status")
async def check_ollama_status():
    """Check if Ollama is running."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags", timeout=5) as response:
                if response.status == 200:
                    return {"status": "running"}
                else:
                    return {"status": "error", "message": f"HTTP error: {response.status}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/webgpu-support")
async def check_webgpu_support():
    """
    This is a placeholder endpoint. WebGPU support is actually checked client-side
    since it's a browser feature. This endpoint just returns information about the
    WebGPU model we're using.
    """
    return {
        "model_name": "DeepSeek-R1-Distill-Qwen-1.5B-ONNX",
        "model_size": "1.5B parameters",
        "model_source": "onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX",
        "requires_webgpu": True,
        "info": "This model runs entirely in your browser using WebGPU"
    }

def clean_llm_output(text):
    """Clean the LLM output by removing <think> tags and extracting Python code."""
    # Remove <think> tags and their content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Remove other common tags that might appear in the output
    text = re.sub(r'<.*?>', '', text, flags=re.DOTALL)
    
    # Try to extract Python code blocks with explicit language markers
    python_blocks = re.findall(r'```(?:python)?(.*?)```', text, re.DOTALL)
    if python_blocks:
        return python_blocks[0].strip()
    
    # Try to extract code blocks without language markers
    code_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # If no code blocks, try to find code by indentation patterns
    lines = text.split('\n')
    code_lines = []
    in_code_block = False
    
    for line in lines:
        # Skip explanatory text at the beginning
        if not in_code_block and (line.strip().startswith('def ') or 
                                 line.strip().startswith('class ') or 
                                 line.strip().startswith('import ') or 
                                 line.strip().startswith('from ') or
                                 line.strip().startswith('# ') or
                                 line.strip().startswith('#!')):
            in_code_block = True
        
        if in_code_block:
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines).strip()
    
    # If all else fails, return the original text with non-code parts removed
    # Remove common non-code parts
    text = re.sub(r'^Here\'s|^Sure|^I\'ll|^This code|^The following|^Here is', '', text, flags=re.IGNORECASE)
    return text.strip()

@app.post("/api/generate")
async def generate_code(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        model = data.get("model", "deepseek-r1")
        
        # Format the prompt for code generation
        formatted_prompt = f"Generate Python code for: {prompt}. Only provide the Python code, no explanations. Make sure the code is complete, well-documented with comments, and follows best practices."
        
        # Call Ollama API
        response = requests.post(
            f"{OLLAMA_API_URL}/generate",
            json={
                "model": model,
                "prompt": formatted_prompt
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Error calling Ollama API")
        
        ollama_response = response.json()
        raw_output = ollama_response.get("response", "")
        
        # Clean and extract the code
        code = clean_llm_output(raw_output)
        
        return {"code": code, "raw_output": raw_output}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    logger.info("WebSocket connection attempt received")
    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted")
        
        # Receive the initial request
        data = await websocket.receive_json()
        prompt = data.get("prompt", "")
        model = data.get("model", "deepseek-r1")
        
        logger.info(f"WebSocket request received: prompt='{prompt[:30]}...' model='{model}'")
        
        # Send initial status
        await websocket.send_json({"status": "starting"})
        
        # Format the prompt for code generation
        formatted_prompt = f"Generate Python code for: {prompt}. Only provide the Python code, no explanations. Make sure the code is complete, well-documented with comments, and follows best practices."
        
        # Set up the streaming request to Ollama
        logger.info(f"Setting up streaming request to Ollama API for model: {model}")
        
        # Debug the Ollama API request
        logger.info(f"Ollama API URL: {OLLAMA_API_URL}/generate")
        logger.info(f"Request payload: {{'model': '{model}', 'prompt': '{formatted_prompt[:30]}...', 'stream': True}}")
        
        # Use direct HTTP request with asyncio for better streaming
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            try:
                # Make the API request with streaming
                async with session.post(
                    f"{OLLAMA_API_URL}/generate",
                    json={
                        "model": model,
                        "prompt": formatted_prompt,
                        "stream": True
                    },
                    timeout=aiohttp.ClientTimeout(total=600)  # 10 minutes timeout
                ) as response:
                    
                    if response.status != 200:
                        error_msg = f"Error: {response.status}"
                        logger.error(f"Ollama API error: {error_msg}")
                        await websocket.send_json({"error": error_msg})
                        return
                    
                    # Stream the response
                    buffer = ""
                    last_update_time = asyncio.get_event_loop().time()
                    
                    # Start a heartbeat task to keep the connection alive
                    async def send_heartbeat():
                        nonlocal last_update_time
                        while True:
                            current_time = asyncio.get_event_loop().time()
                            # Send a heartbeat every 30 seconds if no updates
                            if current_time - last_update_time > 30:
                                try:
                                    logger.info("Sending heartbeat to keep connection alive")
                                    await websocket.send_json({"heartbeat": True, "timestamp": current_time})
                                    last_update_time = current_time
                                except Exception as e:
                                    logger.error(f"Error sending heartbeat: {str(e)}")
                                    break
                            await asyncio.sleep(5)  # Check every 5 seconds
                    
                    # Start the heartbeat task
                    heartbeat_task = asyncio.create_task(send_heartbeat())
                    
                    try:
                        async for line in response.content:
                            line = line.strip()
                            if line:
                                try:
                                    # Parse the JSON chunk
                                    chunk_data = json.loads(line)
                                    logger.info(f"Received chunk: {chunk_data.keys()}")
                                    
                                    # Extract the response text
                                    if "response" in chunk_data:
                                        chunk_text = chunk_data["response"]
                                        buffer += chunk_text
                                        
                                        # Send each chunk immediately to the client
                                        logger.info(f"Sending chunk: '{chunk_text}'")
                                        await websocket.send_json({
                                            "chunk": chunk_text,
                                            "text": buffer,
                                            "done": False
                                        })
                                        
                                        # Update the last update time
                                        last_update_time = asyncio.get_event_loop().time()
                                        
                                        # Force a small delay to ensure client receives updates
                                        await asyncio.sleep(0.01)
                                    
                                    # Check if we're done
                                    if chunk_data.get("done", False):
                                        # Process the final code
                                        clean_code = clean_llm_output(buffer)
                                        logger.info("Code generation completed")
                                        
                                        # Send the final result
                                        await websocket.send_json({
                                            "done": True,
                                            "final_code": clean_code,
                                            "text": buffer
                                        })
                                        break
                                        
                                except json.JSONDecodeError as e:
                                    logger.error(f"JSON decode error: {str(e)} for line: {line}")
                                    await websocket.send_json({"error": f"Error parsing response: {str(e)}"})
                                except Exception as e:
                                    logger.exception(f"Error processing chunk: {str(e)}")
                                    await websocket.send_json({"error": f"Error processing response: {str(e)}"})
                    finally:
                        # Cancel the heartbeat task when done
                        heartbeat_task.cancel()
                        try:
                            await heartbeat_task
                        except asyncio.CancelledError:
                            pass
            except aiohttp.ClientError as e:
                logger.exception(f"aiohttp client error: {str(e)}")
                await websocket.send_json({"error": f"Request error: {str(e)}"})
            except asyncio.TimeoutError:
                logger.exception("Request timed out")
                await websocket.send_json({"error": "Request timed out"})
            except Exception as e:
                logger.exception(f"Unexpected error: {str(e)}")
                await websocket.send_json({"error": f"Unexpected error: {str(e)}"})
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception(f"WebSocket error: {str(e)}")
        # Send error to client
        try:
            if websocket.client_state != "DISCONNECTED":
                await websocket.send_json({"error": str(e)})
        except:
            logger.exception("Failed to send error to client")

@app.websocket("/ws/test")
async def websocket_test(websocket: WebSocket):
    logger.info("Test WebSocket connection attempt received")
    await websocket.accept()
    logger.info("Test WebSocket connection accepted")
    
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received message: {data}")
            await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        logger.info("Test WebSocket disconnected")

if __name__ == "__main__":
    # Debug: Print all registered routes
    for route in app.routes:
        print(f"Route: {route.path}, methods: {getattr(route, 'methods', None)}")
    
    # Get port from environment variable with fallback to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Run directly with uvicorn, binding to 0.0.0.0 to make it accessible
    uvicorn.run(app, host="0.0.0.0", port=port)
    
    logger.info("Server started. Press Ctrl+C to stop.")
    
    try:
        # Wait for the stop event or for the server thread to finish
        while not stop_event.is_set() and server_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        # This is a fallback in case the signal handler doesn't catch it
        logger.info("KeyboardInterrupt detected. Shutting down...")
        stop_event.set()
    
    # If we get here because of Ctrl+C, the signal handler has already set the stop_event
    if stop_event.is_set():
        logger.info("Shutting down server...")
        # Give the server a moment to process any pending requests
        time.sleep(1)
        # Force exit as uvicorn doesn't always exit cleanly
        logger.info("Server shutdown complete.")
        sys.exit(0)
