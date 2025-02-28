# Python Code AI Sandbox with Ollama

This is an interactive web application that uses Ollama to generate Python code based on user prompts. The generated code can be executed directly in the browser using Pyodide.

## Features

- **Multiple AI Models**: Supports various Ollama models for code generation
- **WebGPU Support**: Run AI models directly in the browser using WebGPU
- **Real-time Streaming**: Stream AI responses in real-time using WebSockets
- **Code Execution**: Execute generated Python code directly in the browser
- **Syntax Highlighting**: Beautiful code highlighting with CodeMirror
- Real-time Python code execution using Pyodide
- Dark-themed UI with code editor
- Code editor with syntax highlighting and advanced features

## Requirements

- Python 3.8+
- Ollama installed and running locally (for Ollama models)
- Modern web browser with WebGPU support (for WebGPU models)

## Setup

1. Install Ollama following the instructions at [ollama.ai](https://ollama.ai)

2. Pull the models you want to use (if you haven't already):
```bash
ollama pull deepseek-r1
```

3. Start Ollama:
```bash
ollama serve
```

4. Create a conda environment (recommended):
```bash
conda create -n r1python python=3.10
conda activate r1python
```

5. Install the required dependencies:
```bash
pip install -r requirements.txt
```

6. Run the FastAPI server:
```bash
python main.py
```

7. Open your browser and navigate to `http://localhost:8000`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/fastrocket/r1python.git
   cd r1python
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Start the server:
   ```
   python server.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## Usage

1. Enter a prompt describing the Python code you want to generate
2. Select an AI model from the dropdown
3. Click "Generate" to create the code
4. The generated code will appear in the editor
5. Click "Run" to execute the code directly in the browser

## WebSocket Streaming

The application uses WebSockets to stream responses from the Ollama API in real-time, providing a more interactive experience.

## WebGPU Support

For browsers with WebGPU support, you can run AI models directly in the browser without needing to call external APIs.

### Downloading Models for WebGPU

The application includes a script to download the deepseek-r1 model for WebGPU:

```bash
python download_model.py
```

This will download the model files to the `static/models/deepseek-r1` directory. The model is approximately 2.5GB in size.

**Note:** The large model file (`model.safetensors`) is not included in the repository due to GitHub's file size limitations. You must run the download script to get the complete model.

You can also specify a different output directory:

```bash
python download_model.py --output-dir /path/to/models/deepseek-r1
```

### Using WebGPU Models

1. Select a WebGPU model from the dropdown menu
2. Click the "Load" button for the selected model
3. Wait for the model to load (this may take a minute)
4. Enter a prompt and click "Generate" to create code using the WebGPU model

Note: WebGPU models run entirely in your browser and do not require an internet connection once loaded.

## Troubleshooting

- If you see "Ollama is not running" message, make sure Ollama is started with `ollama serve`
- If no models are shown in the dropdown, ensure you have pulled at least one model with `ollama pull MODEL_NAME`
- If code generation is slow, consider using a smaller model
- For long-running code generation, the application includes a heartbeat mechanism to keep the connection alive

## License

MIT
