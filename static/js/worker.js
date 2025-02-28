// Worker for WebGPU model processing
// This worker handles loading and inference with WebGPU models

// Import from the transformers library
import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  InterruptableStoppingCriteria,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.1/+esm";

// Global variables that will be initialized when the transformers library is imported
let deepseekModel;
let deepseekTokenizer;

let distilBertModel = null;
let distilBertTokenizer = null;
let distilBertPipeline = null;
let isDistilBertLoaded = false;

/**
 * Helper function to perform feature detection for WebGPU
 */
async function detectWebGPU() {
  if (typeof navigator === "undefined") return false;
  return navigator.gpu !== undefined;
}

/**
 * Helper function to perform feature detection for WebGPU
 */
async function check() {
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("WebGPU is not supported (no adapter found)");
    }
    self.postMessage({
      type: "check",
      supported: true,
    });
  } catch (e) {
    self.postMessage({
      type: "error",
      error: e.toString(),
    });
  }
}

/**
 * This class uses the Singleton pattern to enable lazy-loading of the pipeline
 */
class DeepseekPipeline {
  // Use the ONNX model that's specifically prepared for WebGPU
  static model_id = "onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX";
  static isLoading = false;
  static modelSizeGB = 1.28; // Model size in GB

  static async getInstance(progress_callback = null) {
    try {
      if (!this.tokenizer) {
        self.postMessage({
          type: "loading-progress", 
          progress: { 
            status: "Loading tokenizer",
            percent: 0,
            details: "Initializing tokenizer"
          }
        });
        
        console.log("DEBUG - Loading Deepseek tokenizer from:", this.model_id);
        this.tokenizer = await AutoTokenizer.from_pretrained(this.model_id, {
          progress_callback,
        });
        
        self.postMessage({
          type: "loading-progress", 
          progress: { 
            status: "Tokenizer loaded successfully",
            percent: 0.2,
            details: "Tokenizer ready, preparing to load model"
          }
        });
        console.log("DEBUG - Deepseek tokenizer loaded successfully");
      }

      if (!this.model && !this.isLoading) {
        this.isLoading = true;
        console.log("DEBUG - Starting Deepseek model loading process");
        
        self.postMessage({
          type: "loading-progress", 
          progress: { 
            status: "Loading model",
            percent: 0.3,
            details: "Initializing model loading process"
          }
        });
        
        console.log("DEBUG - Loading Deepseek model from:", this.model_id);
        console.log("DEBUG - Using WebGPU backend with q4f16 quantization");
        
        try {
          // Track loading start time
          const startTime = Date.now();
          
          // Custom progress tracker
          let lastFile = '';
          let lastPercent = 0;
          
          const enhancedProgressCallback = (progress) => {
            // Call the original progress callback
            if (progress_callback) {
              progress_callback(progress);
            }
            
            // If this is a file progress update
            if (progress && progress.file) {
              // Extract the file name from the path
              lastFile = progress.file.split('/').pop();
              
              // Format the file name to match the photo (onnx/model_q4f16.onnx)
              if (lastFile.includes('.onnx')) {
                lastFile = `onnx/${lastFile}`;
              }
              
              lastPercent = progress.progress || 0;
              
              // Send a more detailed progress update
              self.postMessage({
                type: "loading-progress",
                progress: {
                  status: "Loading model...",
                  percent: 0.3 + (lastPercent * 0.7), // Scale to 30%-100% range
                  details: `${lastFile} (${Math.round(lastPercent * 100)}% of ${this.modelSizeGB}GB)`
                }
              });
            }
          };
          
          this.model = await AutoModelForCausalLM.from_pretrained(this.model_id, {
            dtype: "q4f16",
            device: "webgpu",
            progress_callback: enhancedProgressCallback,
          });
          
          // Calculate loading time
          const loadTime = ((Date.now() - startTime) / 1000).toFixed(1);
          console.log(`DEBUG - Deepseek model loaded successfully in ${loadTime}s`);
          
          self.postMessage({
            type: "ready",
            model: "deepseek",
            message: `Deepseek-R1 model loaded and ready!`
          });
        } catch (modelError) {
          this.isLoading = false;
          console.error("ERROR loading Deepseek model:", modelError);
          throw modelError;
        }
      }

      return [this.tokenizer, this.model];
    } catch (error) {
      console.error("Error in getInstance:", error);
      throw error;
    }
  }
}

let stoppingCriteria = new InterruptableStoppingCriteria();

/**
 * DistilBertPipeline - handles loading the DistilBERT model
 */
class DistilBertPipeline {
  static model_id = "Xenova/distilbert-base-uncased";
  static tokenizer = null;
  static model = null;
  static pipeline = null;

  static async getInstance(progress_callback = null) {
    try {
      if (!this.tokenizer || !this.model) {
        self.postMessage({
          type: "loading-progress", 
          progress: { 
            status: "Loading DistilBERT components",
            percent: 0,
            details: "Initializing"
          }
        });
        
        // Import the transformers library
        const transformers = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.7.0');
        
        try {
          // Create a fill-mask pipeline directly
          self.postMessage({
            type: "loading-progress", 
            progress: { 
              status: "Creating fill-mask pipeline",
              percent: 0.2,
              details: "This may take a moment..."
            }
          });
          
          // Use pipeline for fill-mask task
          this.pipeline = await transformers.pipeline('fill-mask', this.model_id, {
            progress_callback,
            revision: 'main',
            quantized: false,
          });
          
          // Extract the tokenizer and model from the pipeline
          this.tokenizer = this.pipeline.tokenizer;
          this.model = this.pipeline.model;
          
          self.postMessage({
            type: "loading-progress", 
            progress: { 
              status: "DistilBERT pipeline created successfully",
              percent: 0.9,
              details: "Ready for inference"
            }
          });
        } catch (error) {
          console.error("Error creating DistilBERT pipeline:", error);
          
          // Fallback to loading components separately
          try {
            self.postMessage({
              type: "loading-progress", 
              progress: { 
                status: "Trying fallback loading method",
                percent: 0.3,
                details: "Loading tokenizer..."
              }
            });
            
            // Load tokenizer
            this.tokenizer = await transformers.AutoTokenizer.from_pretrained(this.model_id, {
              progress_callback,
            });
            
            self.postMessage({
              type: "loading-progress", 
              progress: { 
                status: "Loading model...",
                percent: 0.5,
                details: "This may take a moment..."
              }
            });
            
            // Load model - try CPU if WebGPU fails
            try {
              this.model = await transformers.AutoModelForMaskedLM.from_pretrained(this.model_id, {
                progress_callback,
                quantized: false,
              });
            } catch (modelError) {
              console.error("Error loading model:", modelError);
              throw new Error("Failed to load DistilBERT model: " + modelError.message);
            }
          } catch (fallbackError) {
            console.error("Fallback loading failed:", fallbackError);
            throw fallbackError;
          }
        }
        
        self.postMessage({
          type: "ready",
          model: "distilbert"
        });
      }

      return [this.tokenizer, this.model, this.pipeline];
    } catch (error) {
      console.error("Error in DistilBertPipeline.getInstance:", error);
      throw error;
    }
  }
  
  // Helper method to run the fill-mask prediction
  static async predict(text) {
    if (this.pipeline) {
      // If we have a pipeline, use it directly
      return await this.pipeline(text);
    } else if (this.tokenizer && this.model) {
      // Otherwise, manually perform the steps
      const inputs = await this.tokenizer(text, { return_tensors: 'pt' });
      const outputs = await this.model(inputs);
      
      // Extract mask token positions
      const maskTokenId = this.tokenizer.mask_token_id;
      const inputIds = inputs.input_ids[0];
      const maskPositions = [];
      
      for (let i = 0; i < inputIds.length; i++) {
        if (inputIds[i] === maskTokenId) {
          maskPositions.push(i);
        }
      }
      
      // Process predictions for each mask position
      const results = [];
      
      for (const position of maskPositions) {
        const logits = outputs.logits[0][position];
        
        // Get top 5 predictions
        const topk = 5;
        const topkIndices = [...Array(logits.length).keys()]
          .sort((a, b) => logits[b] - logits[a])
          .slice(0, topk);
        
        // Format results
        const predictions = [];
        for (const idx of topkIndices) {
          predictions.push({
            token: idx,
            token_str: await this.tokenizer.decode([idx]),
            score: logits[idx],
          });
        }
        
        results.push(predictions);
      }
      
      return results;
    } else {
      throw new Error("DistilBERT pipeline not initialized");
    }
  }
}

// Define a progress callback function that will be used during model loading
function createProgressCallback() {
  return function(progress) {
    console.log("Progress update:", JSON.stringify(progress));
    
    if (typeof progress === 'number') {
      // Progress is a number between 0 and 1
      self.postMessage({
        type: "loading-progress",
        progress: {
          status: "Loading model files",
          percent: progress,
          details: progress.file ? `Loading ${progress.file}` : null
        }
      });
    } else if (progress && progress.file) {
      // Progress has file information
      const fileName = progress.file.split('/').pop();
      const percentComplete = progress.progress || 0;
      
      self.postMessage({
        type: "loading-progress",
        progress: {
          status: `Loading ${fileName}`,
          percent: percentComplete,
          details: `${fileName} (${Math.round(percentComplete * 100)}%)`
        }
      });
    } else if (progress && progress.status) {
      // Progress has a status message
      self.postMessage({
        type: "loading-progress",
        progress: {
          status: progress.status,
          percent: progress.percent || 0,
          details: progress.details || null
        }
      });
    }
  };
}

async function generateWithDeepseek(prompt) {
  try {
    // Check if model is loaded
    if (!deepseekModel || !deepseekTokenizer) {
      throw new Error("Deepseek model not loaded. Please load the model first.");
    }
    
    // Send an update to start
    self.postMessage({
      type: "update",
      output: "Generating with Deepseek-R1..."
    });
    
    // Make sure the prompt is a string
    if (typeof prompt !== 'string') {
      prompt = String(prompt);
    }
    
    // Store the original prompt before cleaning
    const originalPrompt = prompt;
    
    // Log the prompt for debugging
    console.log("DEBUG - Received prompt:", prompt);
    
    // Clean up repeated prompts - very common issue
    const repeatPromptPattern = /(Generate Python code for:\s*.*?\.(?:\s*Only provide).*?)(?=\1|$)/i;
    if (repeatPromptPattern.test(prompt)) {
      const match = prompt.match(repeatPromptPattern);
      if (match && match[1]) {
        console.log("DEBUG - Detected repeated prompt, cleaning up");
        prompt = match[1];
      }
    }
    
    console.log("DEBUG - Cleaned prompt:", prompt);
    
    // Format the inputs with proper error handling
    let inputs;
    try {
      // Create a messages array like the demo does
      const messages = [
        { role: "user", content: prompt }
      ];
      
      // Use apply_chat_template like the demo
      inputs = await deepseekTokenizer.apply_chat_template(messages, {
        add_generation_prompt: true,
        return_dict: true,
      });
      console.log("DEBUG - Applied chat template successfully");
    } catch (templateError) {
      console.error("Error applying chat template:", templateError);
      // Fall back to direct tokenization
      try {
        inputs = await deepseekTokenizer(prompt, {
          return_tensors: "pt", // Use PyTorch-style tensors
          add_special_tokens: true,
        });
        console.log("DEBUG - Tokenized inputs successfully");
      } catch (tokenizeError) {
        console.error("Error tokenizing prompt:", tokenizeError);
        // Try a simpler approach if the first fails
        inputs = {
          input_ids: await deepseekTokenizer.encode(prompt, {
            return_tensors: "pt",
            add_special_tokens: true,
          })
        };
        console.log("DEBUG - Using fallback tokenization approach");
      }
    }
    
    // Get token IDs for thinking tags to track state
    const thinkTokens = await deepseekTokenizer.encode(
      "<think></think>",
      { add_special_tokens: false }
    );
    
    console.log("DEBUG - Thinking token IDs:", thinkTokens);
    
    // We need to handle the case where the tokens might be split differently
    // So let's also encode just the opening and closing tags separately
    const openThinkToken = await deepseekTokenizer.encode(
      "<think>",
      { add_special_tokens: false }
    );
    
    const closeThinkToken = await deepseekTokenizer.encode(
      "</think>",
      { add_special_tokens: false }
    );
    
    console.log("DEBUG - Open think token:", openThinkToken);
    console.log("DEBUG - Close think token:", closeThinkToken);
    
    // Track all generated outputs - including thinking tokens
    let fullOutput = "";
    let state = "initial"; // Start in initial state, waiting for <think>
    
    // Set up a token callback to detect state changes
    const token_callback_function = (tokens) => {
      const tokenId = tokens[0];
      console.log("DEBUG - Token callback received token:", tokenId);
      
      // Check for state transitions
      if (state === "initial") {
        // Check if this token is part of the opening think tag
        if (openThinkToken.includes(tokenId) || thinkTokens.includes(tokenId)) {
          console.log("DEBUG - Detected opening think token, switching to thinking state");
          state = "thinking";
        }
      } else if (state === "thinking") {
        // Check if this token is part of the closing think tag
        if (closeThinkToken.includes(tokenId) || 
            (thinkTokens.length > 1 && tokenId === thinkTokens[1])) {
          console.log("DEBUG - Detected closing think token, switching to answering state");
          state = "answering";
        }
      }
    };
    
    // Set up a streamer to process tokens as they're generated
    const callback_function = (output) => {
      // Special handling for the first token
      if (fullOutput === "") {
        console.log("DEBUG - First token received:", output);
        
        // If the first token doesn't start with <think>, prepend it
        if (!output.startsWith("<think>")) {
          console.log("DEBUG - First token doesn't start with <think>, prepending it");
          fullOutput = "<think>";
        }
      }
      
      // Accumulate the output (including thinking tokens)
      fullOutput += output;
      console.log("DEBUG - New token output:", output);
      
      // Send update with the full output including thinking tokens and state
      self.postMessage({
        type: "update",
        output: fullOutput,
        state: state
      });
    };
    
    const streamer = new TextStreamer(deepseekTokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function,
      token_callback_function
    });
    
    console.log("DEBUG - Starting model generation with max_new_tokens:", 2048);
    
    // Generate text with the model
    const { sequences } = await deepseekModel.generate({
      ...inputs,
      do_sample: false,  // Use greedy decoding for more predictable outputs
      max_new_tokens: 2048,
      streamer,
      stopping_criteria: stoppingCriteria,
      return_dict_in_generate: true
    });
    
    console.log("DEBUG - Generation complete");
    
    // Decode the final output
    let decoded = await deepseekTokenizer.batch_decode(sequences, {
      skip_special_tokens: true,
    });
    
    console.log("DEBUG - Decoded output:", decoded[0]);
    
    // Get the raw output (with thinking tags)
    let finalOutput = decoded[0] || "";
    
    // Check if we should preserve thinking tokens
    const isCodeOnly = prompt.includes("Only provide the Python code, no explanations");
    
    // If code-only is checked, remove thinking tokens; otherwise keep them
    let cleanedOutput = isCodeOnly ? finalOutput.replace(/<think>([\s\S]*?)<\/think>/g, "") : finalOutput;
    
    console.log("DEBUG - Final cleaned output:", cleanedOutput);
    
    // Send completion message with the cleaned-up output
    self.postMessage({
      type: "complete",
      output: cleanedOutput
    });
    
    return cleanedOutput;
  } catch (error) {
    console.error("Error generating with Deepseek:", error);
    self.postMessage({
      type: "error",
      error: error.message || "Unknown error during Deepseek generation"
    });
    throw error;
  }
}

async function generateWithDistilBert(prompt) {
  try {
    self.postMessage({
      type: "update",
      output: "Processing with DistilBERT model..."
    });
    
    // Create a simplified output for display
    let simplifiedOutput = "# DistilBERT Output\n\n";
    simplifiedOutput += `Input: "${prompt}"\n\n`;
    
    try {
      // Convert lowercase [mask] to uppercase [MASK] for proper tokenization
      const normalizedPrompt = prompt.replace(/\[mask\]/gi, "[MASK]");
      if (prompt !== normalizedPrompt) {
        simplifiedOutput += `Normalized: "${normalizedPrompt}"\n\n`;
      }
      
      console.log("DEBUG - Processing input:", normalizedPrompt);
      
      // Use the pipeline predict method
      const results = await DistilBertPipeline.predict(normalizedPrompt);
      console.log("DEBUG - Prediction results:", results);
      
      if (!results || results.length === 0) {
        simplifiedOutput += "## No Predictions Found\n\n";
        simplifiedOutput += "No [MASK] tokens were found in your input. Make sure to include [MASK] in your text.\n\n";
        simplifiedOutput += "Example: 'The capital of France is [MASK].'\n";
      } else {
        // Check if results is an array of predictions (from pipeline) or array of arrays (from manual method)
        const isPipelineResult = results[0] && typeof results[0].token_str === 'string';
        
        if (isPipelineResult) {
          // Format from pipeline results
          simplifiedOutput += "## Top Predictions\n\n";
          
          // Create a table for better formatting with consistent spacing
          simplifiedOutput += "| Rank | Token       | Confidence |\n";
          simplifiedOutput += "|------|-------------|------------|\n";
          
          for (let i = 0; i < Math.min(5, results.length); i++) {
            const result = results[i];
            const percentage = (result.score * 100).toFixed(2);
            // Ensure token is padded for better alignment (fixed width)
            const tokenStr = result.token_str.length > 10 ? 
                            result.token_str.substring(0, 10) : 
                            result.token_str.padEnd(10, ' ');
            // Use consistent spacing for better monospace alignment
            simplifiedOutput += `| ${(i+1)}     | \`${tokenStr}\` | ${percentage}%   |\n`;
          }
          
          simplifiedOutput += "\n";
          
          // Add example sentence with the top prediction
          if (results.length > 0) {
            const topPrediction = results[0].token_str;
            const filledSentence = normalizedPrompt.replace("[MASK]", topPrediction);
            simplifiedOutput += "### Completed Sentence (with top prediction):\n\n";
            simplifiedOutput += `"${filledSentence}"\n\n`;
          }
        } else {
          // Format from manual method (array of arrays)
          simplifiedOutput += `## Predictions for ${results.length} [MASK] token(s)\n\n`;
          
          for (let maskIndex = 0; maskIndex < results.length; maskIndex++) {
            const maskPredictions = results[maskIndex];
            simplifiedOutput += `### [MASK] #${maskIndex + 1}\n\n`;
            
            // Create a table for better formatting with consistent spacing
            simplifiedOutput += "| Rank | Token       | Confidence |\n";
            simplifiedOutput += "|------|-------------|------------|\n";
            
            for (let i = 0; i < maskPredictions.length; i++) {
              const prediction = maskPredictions[i];
              const percentage = (prediction.score * 100).toFixed(2);
              // Ensure token is padded for better alignment (fixed width)
              const tokenStr = prediction.token_str.length > 10 ? 
                              prediction.token_str.substring(0, 10) : 
                              prediction.token_str.padEnd(10, ' ');
              // Use consistent spacing for better monospace alignment
              simplifiedOutput += `| ${(i+1)}     | \`${tokenStr}\` | ${percentage}%   |\n`;
            }
            
            simplifiedOutput += "\n";
          }
          
          // If there's only one mask, add example sentence with the top prediction
          if (results.length === 1 && results[0].length > 0) {
            const topPrediction = results[0][0].token_str;
            const filledSentence = normalizedPrompt.replace("[MASK]", topPrediction);
            simplifiedOutput += "### Completed Sentence (with top prediction):\n\n";
            simplifiedOutput += `"${filledSentence}"\n\n`;
          }
        }
      }
    } catch (error) {
      console.error("Error in DistilBert processing:", error);
      
      simplifiedOutput += `## Error\n\n`;
      simplifiedOutput += `Error processing with model: ${error.message}\n\n`;
      
      // Add debugging information
      simplifiedOutput += "## Debugging Information\n\n";
      simplifiedOutput += "If you're seeing errors, try the following:\n";
      simplifiedOutput += "1. Make sure to use [MASK] (uppercase) in your input\n";
      simplifiedOutput += "2. Keep your input text relatively short\n";
      simplifiedOutput += "3. Try a simpler sentence structure\n";
    }
    
    // Add a note about DistilBERT's purpose
    simplifiedOutput += "\n## Note on DistilBERT Usage\n\n";
    simplifiedOutput += "DistilBERT is a masked language model designed for understanding text, not generating it. ";
    simplifiedOutput += "It's best used for filling in [MASK] tokens in a sentence.\n\n";
    simplifiedOutput += "### Example Prompts:\n";
    simplifiedOutput += "- \"The capital of France is [MASK].\"\n";
    simplifiedOutput += "- \"I feel [MASK] today.\"\n";
    simplifiedOutput += "- \"Python is a [MASK] programming language.\"\n";
    
    // Send complete message with the simplified output
    self.postMessage({
      type: "complete",
      output: simplifiedOutput,
    });
    
    return simplifiedOutput;
  } catch (error) {
    console.error("Error in generateWithDistilBert:", error);
    self.postMessage({
      type: "error",
      error: error.message || "Unknown error during DistilBERT processing",
    });
    throw error;
  }
}

async function loadDeepseek() {
  try {
    self.postMessage({
      type: "status",
      message: "Loading Deepseek-R1 model..."
    });

    console.log("DEBUG - Starting Deepseek model loading process with timeout protection");
    
    // Add a timeout to prevent infinite loading
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => {
        reject(new Error("Timeout loading Deepseek model (60 seconds)"));
      }, 60000); // 60 second timeout
    });
    
    // Load the pipeline and save it for future use.
    try {
      // Race between the model loading and the timeout
      const loadPromise = DeepseekPipeline.getInstance(createProgressCallback());
      
      const [tokenizer, model] = await Promise.race([loadPromise, timeoutPromise]);

      deepseekModel = model;
      deepseekTokenizer = tokenizer;

      // Signal that the model is ready
      self.postMessage({ 
        type: "ready", 
        model: "deepseek" 
      });
      
      self.postMessage({
        type: "status",
        message: "Deepseek model loaded successfully"
      });
    } catch (loadError) {
      // Reset the loading flag in case of error
      DeepseekPipeline.isLoading = false;
      
      console.error("Error loading Deepseek model:", loadError);
      self.postMessage({ 
        type: "error", 
        error: `Failed to load Deepseek model: ${loadError.message || "Unknown error"}`
      });
    }
  } catch (error) {
    console.error("Error in loadDeepseek function:", error);
    self.postMessage({ 
      type: "error", 
      error: error.message || "Unknown error in loadDeepseek function"
    });
  }
}

async function loadDistilBert() {
  try {
    self.postMessage({
      type: "update",
      output: "Loading DistilBERT model..."
    });
    
    try {
      const [tokenizer, model, pipeline] = await DistilBertPipeline.getInstance(createProgressCallback());
      
      distilBertTokenizer = tokenizer;
      distilBertModel = model;
      distilBertPipeline = pipeline;
      
      self.postMessage({
        type: "ready",
        model: "distilbert"
      });
      
      return true;
    } catch (error) {
      console.error("Error loading DistilBERT model:", error);
      self.postMessage({
        type: "error",
        error: "Failed to load DistilBERT model: " + error.message
      });
      return false;
    }
  } catch (error) {
    console.error("Error in loadDistilBert:", error);
    self.postMessage({
      type: "error",
      error: error.message || "Unknown error loading DistilBERT model"
    });
    return false;
  }
}

async function generate(prompt, modelType) {
  self.postMessage({ type: "start" });
  
  try {
    console.log(`Starting generation with model type: ${modelType}`);
    
    if (modelType === 'distilbert') {
      await generateWithDistilBert(prompt);
    } else if (modelType === 'deepseek') {
      await generateWithDeepseek(prompt);
    } else {
      throw new Error(`Unknown model type: ${modelType}`);
    }
  } catch (error) {
    console.error("Generation error:", error);
    
    // Create a fallback response with the error information
    const errorCode = `
# Error during code generation
# ${error.message}

def main():
    """
    This is a fallback function because an error occurred.
    Error: ${error.message}
    """
    print("An error occurred during code generation")
    return None

if __name__ == "__main__":
    main()
`;
    
    // First send the error
    self.postMessage({ 
      type: "error", 
      error: error.message || "Unknown error during generation" 
    });
    
    // Then send a fallback response
    self.postMessage({
      type: "complete",
      output: errorCode
    });
  }
}

/**
 * Cleans model output to remove any prompt prefixes or unnecessary text
 * @param {string} output - The raw model output
 * @param {string} originalPrompt - The original prompt sent to the model
 * @returns {string} The cleaned output
 */
function cleanModelOutput(output, originalPrompt) {
  // If output is empty, return empty string
  if (!output) return "";
  
  console.log("DEBUG - Cleaning output. Original length:", output.length);
  console.log("DEBUG - First 100 chars of output:", output.substring(0, 100));
  
  // Check for repeated "Generate Python code for:" patterns
  if (output.indexOf("Generate Python code for:") !== output.lastIndexOf("Generate Python code for:")) {
    console.log("DEBUG - Found multiple 'Generate Python code for:' patterns");
    
    // Take everything after the last occurrence
    const lastIndex = output.lastIndexOf("Generate Python code for:");
    if (lastIndex !== -1) {
      const afterLastPrompt = output.substring(lastIndex);
      const codeStart = afterLastPrompt.indexOf("\n");
      if (codeStart !== -1) {
        const cleanedOutput = afterLastPrompt.substring(codeStart + 1);
        console.log("DEBUG - Cleaned by taking content after last prompt pattern");
        return cleanedOutput;
      }
    }
  }
  
  // Use a cleaner regex to find and remove the prompt prefix pattern
  // This handles "Generate Python code for: <user prompt>. Only provide..."
  const codePromptRegex = /^Generate Python code for:\s*(.*?)(?=\.\s*Only provide|\n|$)/i;
  const match = output.match(codePromptRegex);
  
  if (match && match[1] && originalPrompt.includes(match[1])) {
    console.log("DEBUG - Found prompt pattern match:", match[1]);
    // Only apply the pattern removal if we're confident it's a prompt repeat
    // Find where the actual response starts (after the prefix and instructions)
    const prefixEnd = output.indexOf("Only provide");
    if (prefixEnd !== -1) {
      const afterInstructions = output.indexOf("\n", prefixEnd);
      if (afterInstructions !== -1) {
        const cleanedOutput = output.substring(afterInstructions + 1);
        console.log("DEBUG - Cleaned using instruction pattern");
        return cleanedOutput;
      }
    }
  }
  
  // Fall back to just returning the output if we couldn't clean it properly
  console.log("DEBUG - Could not clean output, returning as is");
  return output;
}

// Check WebGPU support
async function checkWebGPUSupport() {
  try {
    // First try to detect WebGPU via navigator
    const hasWebGPU = await detectWebGPU();
    if (!hasWebGPU) {
      return false;
    }
    
    // If we can import transformers.js, WebGPU is likely supported
    const tf = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.7.0');
    return !!tf;
  } catch (error) {
    console.warn('Error checking WebGPU support:', error);
    return false;
  }
}

// Send a status update
function sendStatus(message) {
  self.postMessage({
    type: 'status',
    message
  });
}

// Load transformers in advance
import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.7.0').then(tf => {
  console.log('Transformers preloaded successfully');
}).catch(err => {
  console.error('Failed to preload transformers:', err);
});

// Handle messages from the main thread
self.onmessage = async function(event) {
  try {
    const message = event.data;
    console.log("DEBUG - Worker received message:", message.type);

    if (message.type === "load-distilbert") {
      try {
        const [tokenizer, model, pipeline] = await DistilBertPipeline.getInstance(createProgressCallback());
        
        distilBertTokenizer = tokenizer;
        distilBertModel = model;
        distilBertPipeline = pipeline;
        
        self.postMessage({
          type: "model-loaded",
          model: "distilbert"
        });
      } catch (error) {
        console.error("Error loading DistilBERT model:", error);
        self.postMessage({
          type: "error",
          error: "Failed to load DistilBERT model: " + error.message
        });
      }
    } else if (message.type === 'load-deepseek') {
      await loadDeepseek();
    } else if (message.type === 'generate') {
      await generate(message.prompt, message.model);
    } else if (message.type === 'generate-distilbert') {
      await generateWithDistilBert(message.prompt);
    } else if (message.type === 'check') {
      await check();
    } else if (message.type === 'interrupt') {
      stoppingCriteria.interrupt();
    } else {
      console.warn(`Unknown message type: ${message.type}`);
      self.postMessage({
        type: 'error',
        error: `Unknown message type: ${message.type}`
      });
    }
  } catch (error) {
    console.error("Error handling worker message:", error);
    // Make sure to reset any loading flags
    DeepseekPipeline.isLoading = false;
    
    self.postMessage({
      type: 'error',
      error: `Error in worker: ${error.message || "Unknown error"}`
    });
  }
};
