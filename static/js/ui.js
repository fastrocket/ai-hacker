// Global variables for UI elements
let editor;
let statusElement;
let webgpuStatusElement;
let modelSelect;
let generateButton;
let loadDistilBertButton;
let loadDeepseekButton;
let isGenerating = false;
let pyodideInstance = null;
let streamingWs = null;
let useWebSocket = false;

// Initialize UI elements
function initUI() {
    console.log('Initializing UI elements');
    
    // Initialize UI elements
    statusElement = document.getElementById('status');
    webgpuStatusElement = document.getElementById('webgpu-status');
    modelSelect = document.getElementById('model-select');
    generateButton = document.getElementById('generate-button');
    pythonCodeElement = document.getElementById('python-code');
    
    // Initialize CodeMirror editor if the element exists
    const editorElement = document.getElementById('code-editor');
    if (editorElement) {
        editor = CodeMirror.fromTextArea(editorElement, {
            mode: 'python',
            theme: 'monokai',
            lineNumbers: true,
            matchBrackets: true,
            autoCloseBrackets: true,
            indentUnit: 4,
            tabSize: 4,
            indentWithTabs: false,
            lineWrapping: true,
            extraKeys: {
                'Tab': function(cm) {
                    cm.replaceSelection('    ', 'end');
                }
            }
        });
    }
    
    // No longer call init() here to prevent circular initialization
}

// Initialize the application
async function init() {
    try {
        console.log('Initializing application...');
        
        // Initialize core UI elements
        statusElement = document.getElementById('status');
        modelSelect = document.getElementById('model-select');
        generateButton = document.getElementById('generate-button');
        
        // Default WebGPU support to false
        hasWebGPU = false;
        
        // Check for WebGPU support
        try {
            hasWebGPU = await checkWebGPUSupport();
            console.log('WebGPU support:', hasWebGPU);
            
            // Update WebGPU status
            if (webgpuStatusElement) {
                webgpuStatusElement.textContent = hasWebGPU ? 
                    'WebGPU is supported in your browser!' : 
                    'WebGPU is not supported in your browser.';
                webgpuStatusElement.className = hasWebGPU ? 'status status-good' : 'status status-warning';
            }
        } catch (error) {
            console.error('Error checking WebGPU support:', error);
            hasWebGPU = false;
        }
        
        // Setup model options
        await setupModels();
        
        // Setup event handlers
        if (modelSelect) {
            modelSelect.addEventListener('change', handleModelChange);
        }
        
        if (generateButton) {
            generateButton.addEventListener('click', handleGenerateClick);
        }
        
        // Setup WebSocket toggle
        const websocketToggle = document.getElementById('websocket-toggle');
        if (websocketToggle) {
            // Initialize the useWebSocket value from the toggle
            useWebSocket = websocketToggle.checked;
            console.log('WebSocket streaming initialized to:', useWebSocket);
            
            // Add event listener for changes
            websocketToggle.addEventListener('change', function() {
                useWebSocket = this.checked;
                console.log('WebSocket streaming set to:', useWebSocket);
                statusElement.textContent = useWebSocket ? 
                    'WebSocket streaming enabled for Ollama models' : 
                    'WebSocket streaming disabled for Ollama models';
            });
        }
        
        // Update status
        if (statusElement) {
            statusElement.textContent = 'Ready';
        }
        
        console.log('Application initialized successfully');
    } catch (error) {
        console.error('Initialization error:', error);
        if (statusElement) {
            statusElement.textContent = `Error: ${error.message}`;
        }
    }
}

// Set up the available models
async function setupModels() {
    const modelSelect = document.getElementById('model-select');
    
    if (!modelSelect) {
        console.error('Model select element not found');
        return;
    }
    
    // Clear existing options
    modelSelect.innerHTML = '';
    
    // Check if the WEBGPU model IDs are defined
    // They should be defined in config.js, but let's provide fallbacks if not
    const distilbertId = typeof WEBGPU_DISTILBERT_ID !== 'undefined' ? WEBGPU_DISTILBERT_ID : 'distilbert';
    const deepseekId = typeof WEBGPU_DEEPSEEK_ID !== 'undefined' ? WEBGPU_DEEPSEEK_ID : 'deepseek';
    
    // Add WebGPU model options
    const webgpuOptions = [
        { value: deepseekId, text: 'WebGPU: Deepseek-R1', disabled: !hasWebGPU },
        { value: distilbertId, text: 'WebGPU: DistilBERT', disabled: !hasWebGPU }
    ];
    
    webgpuOptions.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option.value;
        optionElement.textContent = option.text;
        optionElement.disabled = option.disabled;
        modelSelect.appendChild(optionElement);
    });
    
    // Add a separator
    const separator = document.createElement('option');
    separator.disabled = true;
    separator.textContent = '─────────────';
    modelSelect.appendChild(separator);
    
    // Fetch Ollama models
    try {
        const ollamaModels = await fetchOllamaModels();
        
        ollamaModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = `Ollama: ${model}`;
            modelSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error fetching Ollama models:', error);
        
        // Add a fallback option if Ollama is not available
        const fallbackOption = document.createElement('option');
        fallbackOption.value = 'fallback';
        fallbackOption.textContent = 'Ollama models unavailable';
        fallbackOption.disabled = true;
        modelSelect.appendChild(fallbackOption);
    }
}

// Fetch available Ollama models
async function fetchOllamaModels() {
    statusElement.textContent = 'Checking for Ollama models...';
    
    try {
        const response = await fetch('/api/models');
        
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        }
        
        // Get the raw response text first for debugging
        const rawText = await response.text();
        console.log('Raw API response:', rawText);
        
        // Try to parse as JSON
        let data;
        try {
            data = JSON.parse(rawText);
        } catch (parseError) {
            console.error('JSON Parse error:', parseError);
            // Return empty array on parse failure
            return [];
        }
        
        if (data && data.models && Array.isArray(data.models)) {
            console.log('Ollama models:', data.models);
            statusElement.textContent = `Found ${data.models.length} Ollama models`;
            return data.models;
        } else {
            console.warn('Invalid models data format:', data);
            statusElement.textContent = 'No Ollama models found';
            return [];
        }
    } catch (error) {
        console.error('Error fetching Ollama models:', error);
        statusElement.textContent = 'Error fetching Ollama models';
        return [];
    }
}

// Handle model selection change
async function handleModelChange() {
    const selectedModel = modelSelect.value;
    
    console.log('Selected model:', selectedModel);
    
    // Reset the status text
    statusElement.textContent = 'Selected ' + selectedModel;
    
    // Check if the WEBGPU model IDs are defined
    // They should be defined in config.js, but let's provide fallbacks if not
    const distilbertId = typeof WEBGPU_DISTILBERT_ID !== 'undefined' ? WEBGPU_DISTILBERT_ID : 'webgpu-distilbert';
    const deepseekId = typeof WEBGPU_DEEPSEEK_ID !== 'undefined' ? WEBGPU_DEEPSEEK_ID : 'webgpu-deepseek-r1';
    
    // Show or hide WebGPU settings
    const webgpuControls = document.getElementById('webgpu-controls');
    
    if (selectedModel === distilbertId || selectedModel === deepseekId) {
        // This is a WebGPU model
        if (webgpuControls) {
            webgpuControls.style.display = 'block';
        }
        
        // Check if WebGPU is supported
        if (!hasWebGPU) {
            statusElement.textContent = 'WebGPU is not supported in your browser. Please select another model.';
            if (generateButton) {
                generateButton.disabled = true;
            }
            return;
        }
        
        // For WebGPU models, check if the model is loaded
        if (selectedModel === distilbertId) {
            if (!isDistilBertLoaded) {
                statusElement.textContent = 'Loading DistilBERT model...';
                try {
                    await loadDistilBertModel();
                } catch (error) {
                    console.error('Error loading DistilBERT:', error);
                    statusElement.textContent = 'Error loading DistilBERT model.';
                }
            } else {
                statusElement.textContent = 'DistilBERT model ready';
            }
        } else if (selectedModel === deepseekId) {
            if (!isDeepseekLoaded) {
                statusElement.textContent = 'Loading Deepseek-R1 model...';
                try {
                    await loadDeepseekModel();
                } catch (error) {
                    console.error('Error loading Deepseek:', error);
                    statusElement.textContent = 'Error loading Deepseek-R1 model.';
                }
            } else {
                statusElement.textContent = 'Deepseek-R1 model ready';
            }
        }
    } else {
        // This is a server-side model
        if (webgpuControls) {
            webgpuControls.style.display = 'none';
        }
        
        // Enable generate button for server-side models
        if (generateButton) {
            generateButton.disabled = false;
        }
        
        statusElement.textContent = `Selected ${selectedModel} (server-side model)`;
    }
}

// Handle the Generate button click
async function handleGenerateClick() {
    try {
        const promptInput = document.getElementById('prompt-input');
        const rawPrompt = promptInput.value.trim();
        
        if (!rawPrompt) {
            statusElement.textContent = 'Please enter a prompt.';
            return;
        }
        
        // Get formatted prompt using centralized configuration
        const prompt = getCurrentPromptTemplate(rawPrompt);
        
        // Check if we're already generating
        if (isGenerating) {
            statusElement.textContent = 'Already generating code. Please wait.';
            return;
        }
        
        // Mark as generating and disable button
        isGenerating = true;
        generateButton.disabled = true;
        
        // Clear editor and raw output
        if (editor) {
            editor.setValue('');
        }
        if (document.getElementById('raw-output')) {
            document.getElementById('raw-output').textContent = '';
        }
        
        const selectedModel = modelSelect.value;
        
        // Handle WebGPU models
        if (selectedModel === WEBGPU_DISTILBERT_ID) {
            statusElement.textContent = 'Preparing DistilBERT model...';
            
            // Load the model first if needed
            if (!isDistilBertLoaded) {
                statusElement.textContent = 'Loading DistilBERT model first...';
                try {
                    await window.loadDistilBertModel();
                } catch (error) {
                    console.error('Error loading DistilBERT model:', error);
                    statusElement.textContent = 'Error loading model: ' + error.message;
                    isGenerating = false;
                    generateButton.disabled = false;
                    return;
                }
            }
            
            try {
                await window.generateWithWebGPU(prompt);
            } catch (error) {
                console.error('Error using DistilBERT:', error);
                statusElement.textContent = 'Error: ' + error.message;
            } finally {
                isGenerating = false;
                generateButton.disabled = false;
            }
            return;
        } else if (selectedModel === WEBGPU_DEEPSEEK_ID) {
            statusElement.textContent = 'Preparing Deepseek-R1 model...';
            
            // Load the model first if needed
            if (!isDeepseekLoaded) {
                statusElement.textContent = 'Loading Deepseek-R1 model first...';
                try {
                    await window.loadDeepseekModel();
                } catch (error) {
                    console.error('Error loading Deepseek-R1 model:', error);
                    statusElement.textContent = 'Error loading model: ' + error.message;
                    isGenerating = false;
                    generateButton.disabled = false;
                    return;
                }
            }
            
            try {
                await window.generateWithWebGPU(prompt);
            } catch (error) {
                console.error('Error using Deepseek-R1:', error);
                statusElement.textContent = 'Error: ' + error.message;
            } finally {
                isGenerating = false;
                generateButton.disabled = false;
            }
            return;
        }
        
        // Handle server-side models
        statusElement.textContent = `Generating code with ${selectedModel}...`;
        
        try {
            await generateWithStreaming(prompt, selectedModel);
        } catch (error) {
            console.error('Error generating code:', error);
            statusElement.textContent = 'Error: ' + error.message;
        } finally {
            isGenerating = false;
            generateButton.disabled = false;
        }
    } catch (error) {
        console.error('Error in generate function:', error);
        statusElement.textContent = 'Error: ' + error.message;
        isGenerating = false;
        generateButton.disabled = false;
    }
}

// Generate code using server-side streaming
async function generateWithStreaming(prompt, model) {
    try {
        // Clear the outputs first
        document.getElementById('raw-output').textContent = '';
        editor.setValue('');
        
        // Update status
        statusElement.textContent = `Generating code with ${model}...`;
        
        // Check if WebSocket should be used
        if (useWebSocket) {
            await generateWithWebSocket(prompt, model);
            return;
        }
        
        const response = await fetch('/generate_stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt: prompt,
                model: model
            })
        });
        
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Update raw output
        const rawOutput = document.getElementById('raw-output');
        if (rawOutput) {
            rawOutput.textContent = data.code || '';
        }
        
        // Update editor with code
        if (editor) {
            editor.setValue(data.code || '');
        }
        
        // Update status
        statusElement.textContent = 'Code generation complete!';
        
    } catch (error) {
        console.error('Error generating with server:', error);
        statusElement.textContent = 'Error: ' + error.message;
    }
}

// Generate code using WebSocket for streaming
async function generateWithWebSocket(prompt, model) {
    return new Promise((resolve, reject) => {
        try {
            statusElement.textContent = 'Generating code with streaming...';
            
            // Close any existing WebSocket connection
            if (streamingWs && streamingWs.readyState === WebSocket.OPEN) {
                streamingWs.close();
            }
            
            // Create a new WebSocket connection
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.hostname;
            const port = 8001; // Use the WebSocket server port
            const wsUrl = `${protocol}//${host}:${port}/ws/generate`;
            console.log(`Connecting to streaming WebSocket at: ${wsUrl}`);
            
            streamingWs = new WebSocket(wsUrl);
            
            // Clear the editor
            editor.setValue('');
            document.getElementById('raw-output').textContent = '';
            
            // Set up WebSocket event handlers
            streamingWs.onopen = () => {
                console.log('WebSocket connection established for streaming');
                
                // Send the request
                streamingWs.send(JSON.stringify({
                    prompt: prompt,
                    model: model
                }));
            };
            
            streamingWs.onerror = (error) => {
                console.error('WebSocket error:', error);
                statusElement.textContent = 'Error: WebSocket connection failed';
                reject(new Error('WebSocket connection failed'));
            };
            
            streamingWs.onclose = () => {
                console.log('WebSocket connection closed');
                resolve(); // Resolve when connection is closed
            };
            
            let accumulatedText = '';
            
            streamingWs.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log("WebSocket message received:", data);
                    
                    if (data.error) {
                        console.error('Error from server:', data.error);
                        statusElement.textContent = `Error: ${data.error}`;
                        reject(new Error(data.error));
                        return;
                    }
                    
                    // Check for status updates
                    if (data.status) {
                        console.log('Status update:', data.status);
                        statusElement.textContent = `Status: ${data.status}`;
                        return;
                    }
                    
                    // Update the raw output if text is available
                    if (data.text) {
                        document.getElementById('raw-output').textContent = data.text;
                        accumulatedText = data.text;
                    }
                    
                    // If we have a chunk, update the editor
                    if (data.chunk) {
                        // If we don't have accumulated text from the server, add the chunk
                        if (!data.text) {
                            accumulatedText += data.chunk;
                            document.getElementById('raw-output').textContent = accumulatedText;
                        }
                        
                        // Update the editor with cleaned text
                        const cleanedText = window.extractPythonCode(accumulatedText);
                        editor.setValue(cleanedText);
                    }
                    
                    // If we have final code, use it
                    if (data.final_code) {
                        const cleanedCode = window.extractPythonCode(data.final_code);
                        editor.setValue(cleanedCode);
                    }
                    
                    // If we're done, update the status
                    if (data.done) {
                        statusElement.textContent = 'Code generated successfully!';
                        
                        // Close the WebSocket connection
                        streamingWs.close();
                        resolve();
                    }
                } catch (parseError) {
                    console.error('Error parsing WebSocket message:', parseError, 'Raw message:', event.data);
                    statusElement.textContent = 'Error parsing server response';
                    reject(parseError);
                }
            };
            
        } catch (error) {
            console.error('Error generating with streaming:', error);
            statusElement.textContent = 'Error generating with streaming: ' + error.message;
            reject(error);
        }
    });
}

// Initialize Pyodide for running Python code in the browser
async function initPyodide() {
    try {
        if (pyodideInstance) {
            console.log('Pyodide already loaded');
            return pyodideInstance;
        }
        
        console.log('Starting Pyodide load');
        statusElement.textContent = 'Loading Python interpreter...';
        
        // Load pyodide with standard output redirected to console
        pyodideInstance = await loadPyodide({
            stdout: (text) => {
                // Convert to string if it's an array of character codes
                let output = text;
                if (Array.isArray(text) || (typeof text === 'object' && text.constructor === Uint8Array)) {
                    output = String.fromCharCode.apply(null, text);
                }
                console.log('Pyodide stdout:', output);
                return text.length;
            },
            stderr: (text) => {
                // Convert to string if it's an array of character codes
                let output = text;
                if (Array.isArray(text) || (typeof text === 'object' && text.constructor === Uint8Array)) {
                    output = String.fromCharCode.apply(null, text);
                }
                console.error('Pyodide stderr:', output);
                return text.length;
            }
        });
        
        console.log('Pyodide loaded successfully!');
        statusElement.textContent = 'Python interpreter loaded successfully!';
        
        return pyodideInstance;
    } catch (error) {
        console.error('Error loading Pyodide:', error);
        statusElement.textContent = 'Error loading Python interpreter: ' + error.message;
        throw error; // Re-throw for handling in calling functions
    }
}

// Handle run button click
async function handleRunCode() {
    console.log('Run button clicked');
    console.log('Pyodide instance:', pyodideInstance);
    
    if (!pyodideInstance) {
        console.log('Pyodide not loaded, attempting to initialize');
        statusElement.textContent = 'Python interpreter not loaded yet. Attempting to load...';
        try {
            await initPyodide();
        } catch (error) {
            console.error('Failed to initialize Pyodide on demand:', error);
            statusElement.textContent = 'Failed to load Python interpreter: ' + error.message;
            return;
        }
    }
    
    const code = editor.getValue();
    console.log('Code to run:', code);
    
    if (!code.trim()) {
        statusElement.textContent = 'No code to run';
        return;
    }
    
    statusElement.textContent = 'Running code...';
    const outputElement = document.getElementById('output');
    outputElement.textContent = '';
    
    try {
        console.log('Setting up stdout');
        // Redirect stdout to capture print statements
        pyodideInstance.setStdout({
            write: (text) => {
                // Handle text output - convert to string if it's an array of character codes
                let output = text;
                if (Array.isArray(text) || (typeof text === 'object' && text.constructor === Uint8Array)) {
                    // Convert array of character codes to a string
                    output = String.fromCharCode.apply(null, text);
                }
                
                console.log('Stdout:', output);
                outputElement.textContent += output;
                return text.length;
            }
        });
        
        // Redirect stderr to capture error output
        pyodideInstance.setStderr({
            write: (text) => {
                // Handle text output - convert to string if it's an array of character codes
                let output = text;
                if (Array.isArray(text) || (typeof text === 'object' && text.constructor === Uint8Array)) {
                    // Convert array of character codes to a string
                    output = String.fromCharCode.apply(null, text);
                }
                
                console.log('Stderr:', output);
                outputElement.textContent += `\nError: ${output}`;
                return text.length;
            }
        });
        
        // Run the code
        console.log('Running Python code');
        await pyodideInstance.runPythonAsync(code);
        console.log('Code execution completed');
        statusElement.textContent = 'Code executed successfully!';
    } catch (error) {
        console.error('Error running Python code:', error);
        outputElement.textContent += `\nError: ${error.message}`;
        statusElement.textContent = 'Error executing code';
    }
}

// Setup event listeners
function setupEventListeners() {
    // Add event listeners to UI elements
    if (generateButton) {
        generateButton.addEventListener('click', handleGenerateClick);
    }
    
    // Add event listener for run code button
    const runButton = document.getElementById('run-button');
    if (runButton) {
        runButton.addEventListener('click', handleRunCode);
    }
    
    // Add event listener for clear button
    const clearButton = document.getElementById('clear-button');
    if (clearButton) {
        clearButton.addEventListener('click', function() {
            if (editor) {
                editor.setValue('');
            }
            document.getElementById('output').textContent = '';
        });
    }
}

// Document loaded handler is now in index.html
// No need for a duplicate event listener here
