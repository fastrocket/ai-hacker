import json
import os
import subprocess
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.request
import urllib.error
import threading
import socketserver
import websocket
import asyncio
import websockets
import aiohttp
import urllib.parse

print(f"Current working directory: {os.getcwd()}")

# Constants
PORT = 8000
OLLAMA_API_URL = "http://localhost:11434/api"
OLLAMA_CLI_CMD = ["ollama", "list"]
# Use absolute path for static directory
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

class OllamaHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        try:
            # Parse the URL
            parsed_url = urllib.parse.urlparse(self.path)
            path = parsed_url.path
            
            print(f"GET request for {path}")
            
            # Check if this is an API request
            if path.startswith('/api/'):
                self.handle_api(path)
                return
            
            # Serve static files from the static directory
            if path == '/':
                path = '/index.html'
                
            # Remove leading slash completely to avoid os.path.join treating it as absolute
            print(f"Path before normalization: {path}")
            # For Windows, replace all slashes with backslashes to avoid path issues
            normalized_path = path.lstrip('/').replace('/', os.sep)
            print(f"Normalized path: {normalized_path}")
            
            # Use the absolute static directory path
            file_path = os.path.join(STATIC_DIR, normalized_path)
            
            print(f"Looking for file: {file_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Does the file exist? {os.path.exists(file_path)}")
            
            # Check if the file exists
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                self.send_error(404, f'File not found: {file_path}')
                return
                
            # Get the content type based on file extension
            content_type = self.get_content_type(file_path)
            
            # Send the response
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.end_headers()
            
            # Read and send the file content
            with open(file_path, 'rb') as file:
                self.wfile.write(file.read())
                
        except Exception as e:
            print(f"Error in do_GET: {e}")
            self.send_error(500, str(e))
    
    def handle_api(self, path):
        """Handle API requests"""
        try:
            if path == '/api/models':
                # Return a list of available Ollama models
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                try:
                    # Get Ollama models
                    models = self.get_ollama_models()
                    # If no models are found, add a default
                    if not models:
                        models = ['deepseek-r1:latest']
                    # Return just JSON with no additional text
                    self.wfile.write(json.dumps({'models': models}).encode())
                except Exception as e:
                    print(f"Error fetching Ollama models: {e}")
                    # Return an empty list with error message
                    self.wfile.write(json.dumps({'error': str(e), 'models': []}).encode())
            elif path == '/api/generate':
                # This endpoint handles HTTP POST requests
                self.send_error(405, 'Method Not Allowed')
            else:
                # Unknown API endpoint
                self.send_error(404, 'API endpoint not found')
        except Exception as e:
            print(f"Error handling API request: {e}")
            self.send_error(500, str(e))
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            # Parse the URL
            parsed_url = urllib.parse.urlparse(self.path)
            path = parsed_url.path
            
            print(f"POST request for {path}")
            
            # Check content length
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length <= 0:
                self.send_error(400, 'Missing request body')
                return
                
            # Read the request body
            post_data = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(post_data)
            
            if path == '/api/generate':
                self.handle_generate_request(data)
            elif path == '/generate_stream':
                self.handle_generate_stream_request(data)
            else:
                self.send_error(404, 'Endpoint not found')
                
        except Exception as e:
            print(f"Error in do_POST: {e}")
            self.send_error(500, str(e))
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        self.end_headers()
    
    def handle_generate_request(self, data):
        prompt = data.get('prompt', '')
        model = data.get('model', 'deepseek-r1')
        
        try:
            # Format the prompt for code generation
            formatted_prompt = f"Generate Python code for: {prompt}. Only provide the Python code, no explanations. Make sure the code is complete, well-documented with comments, and follows best practices."
            
            # Call Ollama API
            response = self.call_ollama_api(model, formatted_prompt)
            
            # Extract code from response
            code = response.get('response', '')
            
            # If the response contains markdown code blocks, extract the code
            if '```python' in code:
                code = code.split('```python')[1]
                if '```' in code:
                    code = code.split('```')[0]
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'code': code.strip()}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
    
    def handle_models_request(self):
        try:
            # Get available models from Ollama
            models = self.get_ollama_models()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'models': models}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
    
    def handle_status_request(self):
        try:
            # Check if Ollama is running
            status = self.check_ollama_status()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': status}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
    
    def handle_webgpu_support_request(self):
        """Handle request for WebGPU model information"""
        response = {
            "model_name": "DistilBERT (WebGPU)",
            "model_size": "66M parameters",
            "requires_webgpu": True,
            "info": "This model runs entirely in your browser using WebGPU"
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def handle_ollama_status_request(self):
        try:
            # Check if Ollama is running
            status = self.check_ollama_status()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': status}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
    
    def call_ollama_api(self, model, prompt):
        url = f"{OLLAMA_API_URL}/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False  # Explicitly disable streaming for non-streaming requests
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                response_data = response.read().decode('utf-8')
                # Ollama API returns a single JSON object for non-streaming responses
                return json.loads(response_data)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response data: {response_data[:200]}...")  # Print first 200 chars for debugging
            raise Exception(f"Failed to parse Ollama response: {e}")
        except urllib.error.URLError as e:
            print(f"Error calling Ollama API: {e}")
            if hasattr(e, 'reason'):
                raise Exception(f"Failed to reach Ollama server. Reason: {e.reason}")
            elif hasattr(e, 'code'):
                raise Exception(f"Ollama server couldn't fulfill the request. Error code: {e.code}")
            else:
                raise Exception("Unknown error occurred while calling Ollama API")
    
    def get_ollama_models(self):
        """Get available Ollama models"""
        if self.check_ollama_api():
            try:
                # First try using the API
                url = f"{OLLAMA_API_URL}/tags"
                print(f"Fetching Ollama models from: {url}")
                req = urllib.request.Request(url)
                
                with urllib.request.urlopen(req) as response:
                    response_data = response.read().decode('utf-8')
                    print(f"Raw Ollama API response: {response_data[:100]}...")
                    
                    # Strip any trailing data after JSON content
                    # Find the position of the last valid JSON closing bracket
                    last_bracket = response_data.rstrip().rfind('}')
                    if last_bracket > 0:
                        response_data = response_data[:last_bracket+1]
                    
                    data = json.loads(response_data)
                    models = [model['name'] for model in data.get('models', [])]
                    print(f"Extracted models: {models}")
                    return models
            except Exception as e:
                print(f"Error in get_ollama_models API method: {e}")
        # Fallback to CLI if API fails
        try:
            print("Falling back to CLI method")
            result = subprocess.run(OLLAMA_CLI_CMD, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"CLI command failed with return code: {result.returncode}")
                return []
            
            lines = result.stdout.strip().split('\n')
            if len(lines) <= 1:  # Only header, no models
                return []
            
            models = []
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    models.append(model_name)
            
            return models
        except:
            return []
    
    def check_ollama_api(self):
        """Check if Ollama API is running"""
        try:
            # Try to connect to Ollama API
            url = f"{OLLAMA_API_URL}/tags"
            req = urllib.request.Request(url)
            
            with urllib.request.urlopen(req, timeout=2) as response:
                # Just check if we get a valid response
                if response.status == 200:
                    print("Ollama API is running")
                    return True
                return False
        except:
            print("Ollama API is not running or encountered an error")
            return False
    
    def check_ollama_status(self):
        try:
            url = f"{OLLAMA_API_URL}/tags"
            req = urllib.request.Request(url)
            
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.getcode() == 200:
                    return "running"
                else:
                    return "error"
        except:
            return "not_running"
    
    def get_content_type(self, path):
        """Get the content type based on file extension"""
        extension = os.path.splitext(path)[1].lower()
        content_types = {
            '.html': 'text/html',
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.json': 'application/json',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.ico': 'image/x-icon',
            '.wasm': 'application/wasm',
            '.txt': 'text/plain',
        }
        
        return content_types.get(extension, 'application/octet-stream')
    
    def serve_static_file(self, path):
        """Serve static files."""
        try:
            # Get the file path
            file_path = os.path.join(os.getcwd(), path)
            
            # Check if the file exists
            if not os.path.exists(file_path):
                self.send_error(404, f"File not found: {path}")
                return
            
            # Determine the content type based on the file extension
            content_type = self.get_content_type(file_path)
            
            # Open the file and send its contents
            with open(file_path, 'rb') as file:
                self.send_response(200)
                self.send_header('Content-type', content_type)
                self.send_header('Content-Length', str(os.path.getsize(file_path)))
                self.end_headers()
                self.wfile.write(file.read())
        except Exception as e:
            print(f"Error serving static file: {e}")
            self.send_error(500, f"Error serving static file: {str(e)}")
    
    def serve_model_file(self, path):
        """Serve model files."""
        try:
            # Get the file path
            file_path = os.path.join(os.getcwd(), 'static/models', path)
            
            # Check if the file exists
            if not os.path.exists(file_path):
                self.send_error(404, f"File not found: {path}")
                return
            
            # Determine the content type based on the file extension
            content_type = self.get_content_type(file_path)
            
            # Open the file and send its contents
            with open(file_path, 'rb') as file:
                self.send_response(200)
                self.send_header('Content-type', content_type)
                self.send_header('Content-Length', str(os.path.getsize(file_path)))
                # Add CORS headers to allow browser access
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
                self.end_headers()
                self.wfile.write(file.read())
        except Exception as e:
            print(f"Error serving model file: {e}")
            self.send_error(500, f"Error serving model file: {str(e)}")

    def handle_generate_stream_request(self, data):
        prompt = data.get('prompt', '')
        model = data.get('model', 'deepseek-r1')
        
        try:
            # Format the prompt for code generation
            formatted_prompt = f"Generate Python code for: {prompt}. Only provide the Python code, no explanations. Make sure the code is complete, well-documented with comments, and follows best practices."
            
            # Call Ollama API
            response = self.call_ollama_api(model, formatted_prompt, stream=True)
            
            # Extract code from response
            code = response.get('response', '')
            
            # If the response contains markdown code blocks, extract the code
            if '```python' in code:
                code = code.split('```python')[1]
                if '```' in code:
                    code = code.split('```')[0]
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'code': code.strip()}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def call_ollama_api(self, model, prompt, stream=False):
        url = f"{OLLAMA_API_URL}/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                response_data = response.read().decode('utf-8')
                # Ollama API returns a single JSON object for non-streaming responses
                return json.loads(response_data)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response data: {response_data[:200]}...")  # Print first 200 chars for debugging
            raise Exception(f"Failed to parse Ollama response: {e}")
        except urllib.error.URLError as e:
            print(f"Error calling Ollama API: {e}")
            if hasattr(e, 'reason'):
                raise Exception(f"Failed to reach Ollama server. Reason: {e.reason}")
            elif hasattr(e, 'code'):
                raise Exception(f"Ollama server couldn't fulfill the request. Error code: {e.code}")
            else:
                raise Exception("Unknown error occurred while calling Ollama API")

class OllamaWebSocketHandler:
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connection based on the path"""
        print(f"WebSocket connection received with path: {path}")
        
        if path == '/ws/test':
            await self.handle_test_websocket(websocket)
        elif path == '/ws/generate':
            await self.handle_generate_websocket(websocket)
        else:
            print(f"Unknown WebSocket path: {path}, defaulting to test handler")
            await self.handle_test_websocket(websocket)
    
    async def handle_test_websocket(self, websocket):
        """Handle test WebSocket connection"""
        print("Handling test WebSocket connection")
        try:
            # Send a connection confirmation
            await websocket.send(json.dumps({'status': 'connected'}))
            
            # Keep the connection open and respond to messages
            while True:
                try:
                    message = await websocket.recv()
                    print(f"Received test message: {message}")
                    
                    # Echo the message back as JSON
                    await websocket.send(json.dumps({
                        'status': 'received',
                        'message': message
                    }))
                except websockets.exceptions.ConnectionClosed:
                    print("Test WebSocket connection closed")
                    break
        except Exception as e:
            print(f"Error in test WebSocket: {e}")
            try:
                await websocket.send(json.dumps({'error': str(e)}))
            except:
                pass
    
    async def handle_generate_websocket(self, websocket):
        """Handle code generation WebSocket connection"""
        try:
            print("Handling generate WebSocket connection")
            
            # Send a connection confirmation
            await websocket.send(json.dumps({'status': 'connected'}))
            
            # Receive the initial request
            try:
                request = await websocket.recv()
                print(f"Received WebSocket request: {request[:100]}...")
                
                request_data = json.loads(request)
                
                prompt = request_data.get('prompt', '')
                model = request_data.get('model', 'deepseek-r1')
                
                print(f"Processing generation request for model: {model}, prompt: {prompt}")
                
                # Format the prompt for code generation
                formatted_prompt = f"Generate Python code for: {prompt}. Only provide the Python code, no explanations. Make sure the code is complete, well-documented with comments, and follows best practices."
                
                # Send an acknowledgment to the client
                await websocket.send(json.dumps({"status": "processing"}))
                
                # Call Ollama API with streaming
                await self.stream_ollama_response(websocket, model, formatted_prompt)
            except websockets.exceptions.ConnectionClosed:
                print("Generate WebSocket connection closed")
                return
            except json.JSONDecodeError as e:
                print(f"JSON decode error in WebSocket request: {e}")
                await websocket.send(json.dumps({'error': f"Invalid JSON in request: {str(e)}"}))
            
        except Exception as e:
            print(f"Error in generate WebSocket: {e}")
            import traceback
            traceback.print_exc()
            try:
                await websocket.send(json.dumps({'error': str(e)}))
            except:
                pass
    
    async def stream_ollama_response(self, websocket, model, prompt):
        """Stream response from Ollama API to the WebSocket client"""
        try:
            print(f"Streaming Ollama response for model: {model}")
            
            # Prepare the request to Ollama API
            url = "http://localhost:11434/api/generate"
            data = {
                "model": model,
                "prompt": prompt,
                "stream": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    print(f"Ollama API response status: {response.status}")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Ollama API error: {error_text}")
                        await websocket.send(json.dumps({"error": f"Ollama API error: {error_text}"}))
                        return
                    
                    # Process the streaming response
                    accumulated_response = ""
                    
                    # Read the response line by line
                    async for line in response.content:
                        if not line:
                            continue
                        
                        line_text = line.decode('utf-8').strip()
                        print(f"Ollama API line: {line_text[:50]}...")
                        
                        try:
                            chunk = json.loads(line_text)
                            
                            # Extract the text chunk
                            text_chunk = chunk.get('response', '')
                            accumulated_response += text_chunk
                            
                            # Send the chunk to the client
                            await websocket.send(json.dumps({
                                "chunk": text_chunk,
                                "text": accumulated_response,
                                "done": chunk.get('done', False)
                            }))
                            
                            # If this is the last chunk, send a final message
                            if chunk.get('done', False):
                                print("Ollama API streaming complete")
                                await websocket.send(json.dumps({
                                    "final_code": accumulated_response,
                                    "done": True
                                }))
                                break
                            
                        except json.JSONDecodeError as e:
                            print(f"Error parsing Ollama API response: {e}, line: {line_text}")
                            await websocket.send(json.dumps({"error": f"Error parsing Ollama response: {e}"}))
                            continue
        
        except aiohttp.ClientError as e:
            print(f"Ollama API client error: {e}")
            await websocket.send(json.dumps({"error": f"Ollama API client error: {str(e)}"}))
        
        except Exception as e:
            print(f"Error streaming Ollama response: {e}")
            import traceback
            traceback.print_exc()
            await websocket.send(json.dumps({"error": f"Error streaming Ollama response: {str(e)}"}))

def run_websocket_server():
    """Run the WebSocket server in a separate thread"""
    # Create an instance of the handler
    ws_handler = OllamaWebSocketHandler()
    
    async def handler(websocket):
        """WebSocket handler function that routes to the appropriate handler"""
        # In websockets 15.0, we need to extract the path from the request headers
        request_info = websocket.request
        path = request_info.path if hasattr(request_info, 'path') else None
        print(f"WebSocket connection received with path: {path}")
        await ws_handler.handle_websocket(websocket, path)
    
    async def start_server():
        # Start the WebSocket server with the handler function
        port = 8001
        print(f"Starting WebSocket server on port {port}...")
        
        try:
            server = await websockets.serve(handler, "localhost", port)
            print("WebSocket server started successfully")
            await asyncio.Future()  # Run forever
        except Exception as e:
            print(f"Error starting WebSocket server: {e}")
    
    # Run the WebSocket server in a separate thread
    def run():
        asyncio.run(start_server())
    
    thread = threading.Thread(target=run, daemon=True)
    thread.start()

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, OllamaHandler)
    print(f"Starting HTTP server on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    # Change to the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Changed working directory to: {os.getcwd()}")
    print(f"Static directory: {STATIC_DIR}")
    print(f"Does static directory exist? {os.path.exists(STATIC_DIR)}")
    
    # Start WebSocket server
    websocket_thread = threading.Thread(target=run_websocket_server)
    websocket_thread.daemon = True
    websocket_thread.start()
    
    # Start HTTP server
    run_server(PORT)
