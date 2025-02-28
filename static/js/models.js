// Function to extract Python code from the generated text
function extractPythonCode(text) {
    console.log('Extracting Python code from:', text);
    
    // If the text is empty, return empty string
    if (!text) {
        return '';
    }
    
    // Remove thinking tokens if present - handle both with and without newlines
    const thinkRegex = /<think>[\s\S]*?<\/think>/g;
    text = text.replace(thinkRegex, "");
    
    // Also handle case where </think> is followed by a newline
    if (text.startsWith("</think>\n")) {
        text = text.substring("</think>\n".length);
    } else if (text.startsWith("</think>")) {
        text = text.substring("</think>".length);
    }
    
    // Clean up the text by removing common prefixes and suffixes
    text = text.trim()
        .replace(/^Here's a Python code to/i, '')
        .replace(/^Here is a Python code to/i, '')
        .replace(/^Here's the Python code to/i, '')
        .replace(/^Here is the Python code to/i, '')
        .replace(/^Here's the code:/i, '')
        .replace(/^Here is the code:/i, '')
        .replace(/^The code is:/i, '')
        .replace(/^Code:/i, '')
        .replace(/^Python code:/i, '');
    
    // If the text contains the prompt, remove it
    if (text.includes("Write Python code for:")) {
        const parts = text.split("Code:");
        text = parts.length > 1 ? parts[1] : text;
    }
    
    // Try to extract code blocks first
    const codeBlockRegex = /```(?:python)?\s*([\s\S]*?)```/g;
    let match;
    let extractedCode = '';
    
    while ((match = codeBlockRegex.exec(text)) !== null) {
        extractedCode += match[1].trim() + '\n\n';
    }
    
    // If code blocks were found, return them
    if (extractedCode) {
        console.log('Extracted code blocks:', extractedCode);
        return extractedCode.trim();
    }
    
    // If no code blocks found, try to identify Python code by common patterns
    // Look for lines that start with common Python patterns
    const lines = text.split('\n');
    const pythonLines = lines.filter(line => {
        const trimmedLine = line.trim();
        return (
            trimmedLine.startsWith('import ') ||
            trimmedLine.startsWith('from ') ||
            trimmedLine.startsWith('def ') ||
            trimmedLine.startsWith('class ') ||
            trimmedLine.startsWith('if ') ||
            trimmedLine.startsWith('for ') ||
            trimmedLine.startsWith('while ') ||
            trimmedLine.startsWith('print(') ||
            trimmedLine.match(/^[a-zA-Z_][a-zA-Z0-9_]* = /) ||
            trimmedLine.startsWith('# ')
        );
    });
    
    if (pythonLines.length > 0) {
        // If we found Python-like lines, return them
        extractedCode = pythonLines.join('\n');
        console.log('Extracted Python-like lines:', extractedCode);
        return extractedCode;
    }
    
    // If all else fails, just return the cleaned text
    console.log('Returning cleaned text:', text);
    return text;
}

// Global variables for model state
let isDistilBertLoaded = false;
let isDeepseekLoaded = false;
let webGPUWorker = null;
let seenFirstThink = false;

/**
 * Initializes the WebGPU worker and sets up the message handling
 */
function initWebGPUWorker() {
    if (webGPUWorker) {
        console.log("WebGPU worker already initialized");
        return;
    }
    
    try {
        console.log("Initializing WebGPU worker");
        
        // Create a new worker
        webGPUWorker = new Worker('/js/worker.js', { type: 'module' });
        
        // Set up the message listener
        webGPUWorker.onmessage = function(e) {
            const data = e.data;
            console.log("Main thread received worker message:", data.type);
            
            switch (data.type) {
                case 'status':
                    // Update status display
                    if (statusElement) {
                        statusElement.textContent = data.message;
                    }
                    break;
                    
                case 'loading':
                    // Handle loading messages
                    if (statusElement) {
                        statusElement.textContent = data.data;
                    }
                    break;
                    
                case 'loading-progress':
                    // Handle loading progress updates
                    const progress = data.progress;
                    console.log("Loading progress update:", progress);
                    
                    // Get loading indicator elements
                    const loadingContainer = document.getElementById('loading-container');
                    const loadingText = document.getElementById('loading-text');
                    const loadingBar = document.getElementById('loading-bar');
                    const loadingDetails = document.getElementById('loading-details');
                    
                    // Show the loading container
                    if (loadingContainer) {
                        loadingContainer.style.display = 'block';
                    }
                    
                    if (progress) {
                        // Update loading indicator elements
                        if (loadingText && progress.status) {
                            loadingText.textContent = progress.status;
                        }
                        
                        if (loadingBar && progress.percent !== undefined) {
                            const percent = typeof progress.percent === 'number' ? progress.percent * 100 : 0;
                            loadingBar.style.width = `${Math.min(Math.max(percent, 0), 100)}%`;
                        }
                        
                        if (loadingDetails && progress.details) {
                            loadingDetails.textContent = progress.details;
                        }
                        
                        // If loading is complete, hide the loading container after a delay
                        if (progress.percent >= 0.99) {
                            // Set the loading bar to exactly 100% for visual completion
                            if (loadingBar) {
                                loadingBar.style.width = '100%';
                            }
                            
                            // Clear any previous text in the loading details
                            if (loadingDetails) {
                                loadingDetails.textContent = 'Ready for inference';
                            }
                            
                            // Update the status text
                            if (loadingText) {
                                loadingText.textContent = 'DistilBERT model loaded successfully';
                            }
                            
                            // Update the main status element
                            if (statusElement) {
                                statusElement.textContent = 'DistilBERT model loaded and ready!';
                            }
                        }
                    }
                    break;
                
                case 'ready':
                    // Handle model ready message
                    if (data.model === 'distilbert') {
                        isDistilBertLoaded = true;
                    } else if (data.model === 'deepseek') {
                        isDeepseekLoaded = true;
                    }
                    break;
                
                case 'error':
                    // Handle error messages
                    console.error("Worker error:", data.error);
                    if (statusElement) {
                        statusElement.textContent = `Error: ${data.error}`;
                    }
                    // Reset model loading flags in case of error
                    if (data.error.includes('Deepseek')) {
                        isDeepseekLoaded = false;
                    } else if (data.error.includes('DistilBERT')) {
                        isDistilBertLoaded = false;
                    }
                    break;
                    
                case 'update':
                    // Handle update messages (e.g. generation progress)
                    // Update raw output display
                    const rawOutput = document.getElementById('raw-output');
                    if (rawOutput) {
                        rawOutput.textContent = data.output || '';
                    }
                    
                    // Track the current output and state
                    const currentOutput = data.output || '';
                    const currentState = data.state || 'initial';
                    
                    console.log("DEBUG - Update message received, state:", currentState, "output length:", currentOutput.length);
                    
                    // Check if this is the first update and it starts with <think>
                    if (!window.seenFirstThink && currentOutput.startsWith("<think>")) {
                        console.log("DEBUG - First <think> tag detected");
                        window.seenFirstThink = true;
                        // Force state to thinking since we found the tag
                        data.state = 'thinking';
                    }
                    
                    // Check for the opening <think> tag in the output if we're in initial state
                    if (currentState === 'initial' && currentOutput.includes('<think>')) {
                        console.log("DEBUG - Found <think> tag in output, switching to thinking state");
                        // Force state to thinking since we found the tag
                        data.state = 'thinking';
                    }
                    
                    // Check for the closing </think> tag if we're in thinking state
                    if (currentState === 'thinking' && currentOutput.includes('</think>')) {
                        console.log("DEBUG - Found </think> tag in output, switching to answering state");
                        // Force state to answering since we found the closing tag
                        data.state = 'answering';
                        
                        // Set the answer index at the position after </think>
                        const thinkEndPos = currentOutput.indexOf('</think>') + '</think>'.length;
                        window.answerIndex = thinkEndPos;
                        
                        console.log("DEBUG - Setting answer index at:", thinkEndPos);
                    }
                    
                    // Store the answer index when state changes from thinking to answering
                    if (!window.answerIndex && data.state === 'answering') {
                        console.log("DEBUG - State changed to answering, setting answer index at:", currentOutput.length);
                        window.answerIndex = currentOutput.length;
                        
                        // Add a visual indicator in the raw output to show where thinking ends
                        if (rawOutput) {
                            // Create a marker element
                            const marker = document.createElement('div');
                            marker.className = 'state-transition-marker';
                            marker.textContent = '--- End of thinking, start of answer ---';
                            marker.style.color = '#4CAF50';
                            marker.style.fontWeight = 'bold';
                            marker.style.borderTop = '1px solid #4CAF50';
                            marker.style.borderBottom = '1px solid #4CAF50';
                            marker.style.padding = '4px';
                            marker.style.margin = '8px 0';
                            
                            // Insert the marker after the raw output
                            rawOutput.parentNode.insertBefore(marker, rawOutput.nextSibling);
                        }
                    }
                    
                    // For streaming updates, show the raw output in the editor
                    if (editor) {
                        // Always show the full output during generation
                        if (data.state === 'initial' || data.state === 'thinking') {
                            // During thinking state, show the full output
                            editor.setValue(currentOutput || '# Thinking...');
                        } else if (data.state === 'answering') {
                            // In answering state, show the full output
                            editor.setValue(currentOutput || '');
                        }
                    }
                    break;
                    
                case 'complete':
                    // Handle completion messages
                    if (generateButton) {
                        generateButton.disabled = false;
                    }
                    
                    // Debug logging
                    console.log("DEBUG - Complete message received, output:", data.output);
                    console.log("DEBUG - Output length:", data.output ? data.output.length : 0);
                    
                    // Update raw output with final result if available
                    if (data.output) {
                        const rawOutput = document.getElementById('raw-output');
                        if (rawOutput) {
                            console.log("DEBUG - Raw output element found, updating content");
                            rawOutput.textContent = data.output;
                        } else {
                            console.error("DEBUG - Raw output element not found!");
                        }
                        
                        // Also update editor if available - extract code if needed
                        if (editor) {
                            console.log("DEBUG - Editor found, updating content");
                            // Check if this is DistilBert output
                            const isDistilBertOutput = data.output.includes('# DistilBERT Output');
                            
                            if (isDistilBertOutput) {
                                // For DistilBert, show the formatted output in the editor
                                console.log("DEBUG - DistilBert output detected, setting to editor");
                                editor.setValue(data.output);
                            } else {
                                // For other models, try to extract Python code
                                let codeToExtract = data.output;
                                
                                // If we have an answer index, only extract from the answer part
                                if (window.answerIndex) {
                                    codeToExtract = data.output.substring(window.answerIndex);
                                }
                                
                                // Extract Python code from the output
                                const extractedCode = extractPythonCode(codeToExtract);
                                
                                // Update the editor with the extracted code
                                editor.setValue(extractedCode || codeToExtract);
                            }
                        } else {
                            console.error("DEBUG - Editor not found!");
                        }
                    }
                    
                    // Remove the state transition marker if it exists
                    const marker = document.querySelector('.state-transition-marker');
                    if (marker) {
                        marker.parentNode.removeChild(marker);
                    }
                    
                    // Reset the answerIndex for the next generation
                    window.answerIndex = null;
                    window.seenFirstThink = false;
                    
                    break;
            }
        };
        
        // Handle worker errors
        webGPUWorker.onerror = function(error) {
            console.error("WebGPU worker error:", error);
            if (statusElement) {
                statusElement.textContent = `Worker error: ${error.message || "Unknown error"}`;
            }
            
            // Reset worker and model loading flags
            webGPUWorker = null;
            isDistilBertLoaded = false;
            isDeepseekLoaded = false;
        };
        
        console.log("WebGPU worker initialized successfully");
    } catch (error) {
        console.error("Error initializing WebGPU worker:", error);
        if (statusElement) {
            statusElement.textContent = `Failed to initialize WebGPU worker: ${error.message}`;
        }
        webGPUWorker = null;
    }
}

// WebGPU model loading functions
async function loadDistilBertModel() {
    if (isDistilBertLoaded) {
        statusElement.textContent = 'DistilBERT model already loaded';
        return;
    }
    
    try {
        // Update status
        statusElement.textContent = 'Loading DistilBERT model...';
        
        // Show loading indicator
        const loadingContainer = document.getElementById('loading-container');
        const loadingText = document.getElementById('loading-text');
        const loadingBar = document.getElementById('loading-bar');
        const loadingDetails = document.getElementById('loading-details');
        
        if (loadingContainer) {
            loadingContainer.style.display = 'block';
        }
        
        if (loadingText) {
            loadingText.textContent = 'Loading DistilBERT model...';
        }
        
        if (loadingBar) {
            loadingBar.style.width = '0%';
        }
        
        if (loadingDetails) {
            loadingDetails.textContent = 'Initializing...';
        }
        
        // Request the worker to load the model
        webGPUWorker.postMessage({ type: 'load-distilbert' });
        
        // Show loading indicator
        return new Promise((resolve) => {
            const checkLoaded = (e) => {
                if (e.data.type === 'ready' && e.data.model === 'distilbert') {
                    // Model loaded successfully
                    isDistilBertLoaded = true;
                    statusElement.textContent = 'DistilBERT model loaded and ready!';
                    
                    // Hide loading indicator
                    const loadingContainer = document.getElementById('loading-container');
                    if (loadingContainer) {
                        loadingContainer.style.display = 'none';
                    }
                    
                    webGPUWorker.removeEventListener('message', checkLoaded);
                    resolve();
                } else if (e.data.type === 'error') {
                    // Model loading failed
                    statusElement.textContent = 'Failed to load DistilBERT model: ' + e.data.error;
                    
                    // Hide loading indicator
                    const loadingContainer = document.getElementById('loading-container');
                    if (loadingContainer) {
                        loadingContainer.style.display = 'none';
                    }
                    
                    webGPUWorker.removeEventListener('message', checkLoaded);
                    resolve();
                }
            };
            
            webGPUWorker.addEventListener('message', checkLoaded);
        });
    } catch (error) {
        console.error('Error loading DistilBERT model:', error);
        statusElement.textContent = 'Error loading DistilBERT model: ' + error.message;
    }
}

async function loadDeepseekModel() {
    if (isDeepseekLoaded) {
        statusElement.textContent = 'Deepseek-R1 model already loaded';
        return;
    }
    
    try {
        // Update status
        statusElement.textContent = 'Loading Deepseek-R1 model...';
        
        // Update WebGPU status if available
        const webgpuStatusElement = document.getElementById('webgpu-status');
        if (webgpuStatusElement) {
            webgpuStatusElement.textContent = 'Loading Deepseek-R1 model...';
            webgpuStatusElement.className = 'status status-loading';
        }
        
        // Show loading indicator
        const loadingContainer = document.getElementById('loading-container');
        const loadingText = document.getElementById('loading-text');
        const loadingBar = document.getElementById('loading-bar');
        const loadingDetails = document.getElementById('loading-details');
        
        if (loadingContainer) {
            loadingContainer.style.display = 'block';
        }
        
        if (loadingText) {
            loadingText.textContent = 'Loading Deepseek-R1 model...';
        }
        
        if (loadingBar) {
            loadingBar.style.width = '0%';
        }
        
        if (loadingDetails) {
            loadingDetails.textContent = 'Initializing...';
        }
        
        // Make sure the worker is initialized
        if (!webGPUWorker) {
            console.log("Initializing WebGPU worker for Deepseek model");
            initWebGPUWorker();
        }
        
        // Request the worker to load the model
        webGPUWorker.postMessage({ type: 'load-deepseek' });
        
        // Show loading indicator and add a timeout
        return new Promise((resolve, reject) => {
            // Add a timeout to prevent hanging
            const timeoutId = setTimeout(() => {
                webGPUWorker.removeEventListener('message', checkLoaded);
                isDeepseekLoaded = false;
                statusElement.textContent = 'Timeout loading Deepseek-R1 model (60 seconds)';
                console.error("Timeout loading Deepseek model");
                
                // Update WebGPU status
                const webgpuStatusElement = document.getElementById('webgpu-status');
                if (webgpuStatusElement) {
                    webgpuStatusElement.textContent = 'Timeout loading Deepseek-R1 model (60 seconds)';
                    webgpuStatusElement.className = 'status status-bad';
                }
                
                // Hide loading indicator
                const loadingContainer = document.getElementById('loading-container');
                if (loadingContainer) {
                    loadingContainer.style.display = 'none';
                }
                
                // Try to terminate and recreate the worker
                if (webGPUWorker) {
                    try {
                        webGPUWorker.terminate();
                    } catch (e) {
                        console.error("Error terminating worker:", e);
                    }
                    webGPUWorker = null;
                }
                
                reject(new Error("Timeout loading Deepseek model"));
            }, 60000); // 60 second timeout
            
            const checkLoaded = (e) => {
                if (e.data.type === 'ready' && e.data.model === 'deepseek') {
                    // Model loaded successfully
                    clearTimeout(timeoutId);
                    isDeepseekLoaded = true;
                    
                    // Update WebGPU status
                    const webgpuStatusElement = document.getElementById('webgpu-status');
                    if (webgpuStatusElement) {
                        webgpuStatusElement.className = 'status status-good';
                    }
                    
                    // Hide loading indicator
                    const loadingContainer = document.getElementById('loading-container');
                    if (loadingContainer) {
                        loadingContainer.style.display = 'none';
                    }
                    
                    webGPUWorker.removeEventListener('message', checkLoaded);
                    resolve();
                } else if (e.data.type === 'error') {
                    // Model loading failed
                    clearTimeout(timeoutId);
                    isDeepseekLoaded = false;
                    statusElement.textContent = 'Failed to load Deepseek-R1 model: ' + e.data.error;
                    
                    // Update WebGPU status
                    const webgpuStatusElement = document.getElementById('webgpu-status');
                    if (webgpuStatusElement) {
                        webgpuStatusElement.textContent = 'Failed to load Deepseek-R1 model: ' + e.data.error;
                        webgpuStatusElement.className = 'status status-bad';
                    }
                    
                    // Hide loading indicator
                    const loadingContainer = document.getElementById('loading-container');
                    if (loadingContainer) {
                        loadingContainer.style.display = 'none';
                    }
                    
                    webGPUWorker.removeEventListener('message', checkLoaded);
                    reject(new Error(e.data.error));
                }
            };
            
            webGPUWorker.addEventListener('message', checkLoaded);
        }).catch(error => {
            console.error('Error in Deepseek model loading promise:', error);
            statusElement.textContent = 'Error loading Deepseek-R1 model: ' + error.message;
            
            // Update WebGPU status
            const webgpuStatusElement = document.getElementById('webgpu-status');
            if (webgpuStatusElement) {
                webgpuStatusElement.textContent = 'Error loading Deepseek-R1 model: ' + error.message;
                webgpuStatusElement.className = 'status status-bad';
            }
            
            // Hide loading indicator
            const loadingContainer = document.getElementById('loading-container');
            if (loadingContainer) {
                loadingContainer.style.display = 'none';
            }
            
            // Ensure we reset the loading state
            isDeepseekLoaded = false;
        });
    } catch (error) {
        console.error('Error loading Deepseek-R1 model:', error);
        statusElement.textContent = 'Error loading Deepseek-R1 model: ' + error.message;
        
        // Update WebGPU status
        const webgpuStatusElement = document.getElementById('webgpu-status');
        if (webgpuStatusElement) {
            webgpuStatusElement.textContent = 'Error loading Deepseek-R1 model: ' + error.message;
            webgpuStatusElement.className = 'status status-bad';
        }
        
        // Hide loading indicator
        const loadingContainer = document.getElementById('loading-container');
        if (loadingContainer) {
            loadingContainer.style.display = 'none';
        }
        
        isDeepseekLoaded = false;
    }
}

// Function to check WebGPU support
async function checkWebGPUSupport() {
    try {
        // Initialize worker which will check WebGPU support
        if (!webGPUWorker) {
            initWebGPUWorker();
        }
        
        // If we already have navigator.gpu, it's supported
        if (navigator.gpu) {
            return true;
        }
        
        // Otherwise ask the worker to check (which will do the more thorough check)
        return new Promise((resolve) => {
            webGPUWorker.postMessage({ type: 'check' });
            
            const checkHandler = (e) => {
                if (e.data.type === 'check') {
                    webGPUWorker.removeEventListener('message', checkHandler);
                    resolve(e.data.supported);
                }
            };
            
            webGPUWorker.addEventListener('message', checkHandler);
            
            // Set a timeout in case the check doesn't complete
            setTimeout(() => {
                webGPUWorker.removeEventListener('message', checkHandler);
                resolve(false);
            }, 5000);
        });
    } catch (error) {
        console.error('Error checking WebGPU support:', error);
        return false;
    }
}

// Generate using WebGPU
async function generateWithWebGPU(prompt) {
    try {
        if (!webGPUWorker) {
            throw new Error("WebGPU worker not initialized.");
        }
        
        const selectedModel = modelSelect.value;
        
        if (!prompt) {
            throw new Error("Please enter a prompt.");
        }
        
        // Log the original prompt
        console.log("DEBUG - Original prompt before formatting:", prompt);
        
        // Get the appropriate prompt template based on UI settings
        prompt = getCurrentPromptTemplate(prompt);
        
        // Log the formatted prompt
        console.log("DEBUG - Formatted prompt:", prompt);
        
        // Disable generate button during generation
        if (generateButton) {
            generateButton.disabled = true;
        }
        
        // Determine which model to use based on selection
        if (selectedModel === WEBGPU_DISTILBERT_ID) {
            if (!isDistilBertLoaded) {
                // Try to load the model first
                statusElement.textContent = 'Loading DistilBERT model...';
                await loadDistilBertModel();
            }
            
            statusElement.textContent = 'Generating with DistilBERT...';
            
            console.log('Generating with DistilBERT model');
            console.log(`Prompt: ${prompt}`);
            
            // Send the generation request to the worker
            webGPUWorker.postMessage({
                type: 'generate-distilbert',
                prompt: prompt
            });
        } else if (selectedModel === WEBGPU_DEEPSEEK_ID) {
            if (!isDeepseekLoaded) {
                // Try to load the model first
                statusElement.textContent = 'Loading Deepseek model...';
                await loadDeepseekModel();
            }
            
            statusElement.textContent = 'Generating with Deepseek-R1...';
            
            console.log('Generating with Deepseek model');
            console.log(`Prompt: ${prompt}`);
            
            // Send the generation request to the worker
            webGPUWorker.postMessage({
                type: 'generate',
                prompt: prompt,
                model: 'deepseek'
            });
        } else {
            throw new Error(`Unsupported WebGPU model: ${selectedModel}`);
        }
        
        // The worker will handle the response asynchronously
    } catch (error) {
        console.error('Error generating with WebGPU:', error);
        statusElement.textContent = `Error: ${error.message}`;
        
        // Re-enable generate button
        if (generateButton) {
            generateButton.disabled = false;
        }
    }
}

// Export functions to the window object
window.generateWithWebGPU = generateWithWebGPU;
window.checkWebGPUSupport = checkWebGPUSupport;
window.loadDistilBertModel = loadDistilBertModel;
window.loadDeepseekModel = loadDeepseekModel;

// Handle model change and update status
function handleModelChange() {
    if (!modelSelect) return;
    
    const selectedModel = modelSelect.value;
    
    // Enable generation button by default
    if (generateButton) {
        generateButton.disabled = false;
    }
    
    // Handle WebGPU models
    if (selectedModel === WEBGPU_DISTILBERT_ID) {
        if (isDistilBertLoaded) {
            statusElement.textContent = 'DistilBERT model loaded and ready';
        } else {
            statusElement.textContent = 'DistilBERT model not loaded. Will load on first use.';
        }
    } else if (selectedModel === WEBGPU_DEEPSEEK_ID) {
        if (isDeepseekLoaded) {
            statusElement.textContent = 'Deepseek-R1 model loaded and ready';
        } else {
            statusElement.textContent = 'Deepseek-R1 model not loaded. Will load on first use.';
        }
    } else {
        // Server-side models are always ready
        statusElement.textContent = `Server model ${selectedModel} selected`;
    }
}
