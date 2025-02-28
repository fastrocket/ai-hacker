// Constants
const WEBGPU_DISTILBERT_ID = "webgpu-distilbert";
const WEBGPU_DEEPSEEK_ID = "webgpu-deepseek-r1";
const DISTILBERT_MODEL_PATH = "/models/distilbert";
const DEEPSEEK_MODEL_PATH = "/models/deepseek-r1";

// Configuration constants for the application

// Ollama API endpoint
const OLLAMA_API_ENDPOINT = "/api/ollama";

// Prompt templates
const PROMPT_TEMPLATES = {
    // Template for code-only prompts
    CODE_ONLY: (userPrompt) => formatPythonCodePrompt(userPrompt),
    
    // Template for regular prompts (just passes through the user's prompt)
    REGULAR: (userPrompt) => userPrompt
};

// Function to format Python code prompt - centralizes the "Generate Python code for" pattern
function formatPythonCodePrompt(userPrompt) {
    console.log("DEBUG - Formatting Python code prompt:", userPrompt);
    // Always return the user's prompt as is
    return userPrompt;
}

// Get the current prompt template based on the Code Only checkbox
function getCurrentPromptTemplate(rawPrompt) {
    // Always use the regular template
    return PROMPT_TEMPLATES.REGULAR(rawPrompt);
}

// Utility functions
function extractPythonCode(text) {
    let extractedCode = '';
    
    // Try to extract Python code blocks
    const pythonCodeBlockRegex = /```(?:python)?\s*([\s\S]*?)```/g;
    let match;
    
    while ((match = pythonCodeBlockRegex.exec(text)) !== null) {
        extractedCode += match[1].trim() + '\n\n';
    }
    
    // If no code blocks found, return the original text with some cleanup
    if (!extractedCode) {
        // Remove markdown formatting
        return text.replace(/\*\*/g, '')
                   .replace(/\*/g, '')
                   .replace(/^#.*$/gm, '')
                   .trim();
    }
    
    return extractedCode.trim();
}

// Helper function to generate sample code based on the prompt
function generateSampleCode(prompt) {
    // Convert prompt to lowercase for easier matching
    const lowerPrompt = prompt.toLowerCase();
    
    // Generate different examples based on the prompt content
    if (lowerPrompt.includes('fibonacci')) {
        return `def fibonacci(n):
    """
    Calculate the Fibonacci sequence up to the nth term.
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib

# Example usage
result = fibonacci(10)
print(result)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]`;
    } else if (lowerPrompt.includes('sort') || lowerPrompt.includes('array')) {
        return `def bubble_sort(arr):
    """
    Implement bubble sort algorithm.
    """
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Example usage
numbers = [64, 34, 25, 12, 22, 11, 90]
sorted_numbers = bubble_sort(numbers)
print(sorted_numbers)  # [11, 12, 22, 25, 34, 64, 90]`;
    } else if (lowerPrompt.includes('flask') || lowerPrompt.includes('web') || lowerPrompt.includes('api')) {
        return `from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample data
tasks = [
    {'id': 1, 'title': 'Learn Python', 'done': True},
    {'id': 2, 'title': 'Learn Flask', 'done': False}
]

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': tasks})

@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = next((task for task in tasks if task['id'] == task_id), None)
    if task:
        return jsonify({'task': task})
    return jsonify({'error': 'Task not found'}), 404

@app.route('/api/tasks', methods=['POST'])
def create_task():
    if not request.json or 'title' not in request.json:
        return jsonify({'error': 'Title is required'}), 400
    
    task = {
        'id': tasks[-1]['id'] + 1 if tasks else 1,
        'title': request.json['title'],
        'done': False
    }
    tasks.append(task)
    return jsonify({'task': task}), 201

if __name__ == '__main__':
    app.run(debug=True)`;
    } else {
        return `def hello_world():
    """
    A simple hello world function.
    """
    print("Hello, World!")

hello_world()

# Here's a more complex example based on your prompt:
def process_data(data):
    """
    Process the given data and return the result.
    """
    result = []
    for item in data:
        if isinstance(item, (int, float)):
            result.append(item * 2)
        elif isinstance(item, str):
            result.append(item.upper())
        else:
            result.append(str(item))
    return result

# Example usage
sample_data = [1, 2.5, "hello", True, None]
processed = process_data(sample_data)
print(processed)  # [2, 5.0, "HELLO", "True", "None"]`;
    }
}
