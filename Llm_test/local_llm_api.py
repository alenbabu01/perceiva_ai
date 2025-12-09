from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Ollama server URL (default port 11434)
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Use the exact model name from `ollama list`: "qwen3:8b"
MODEL_NAME = "qwen3:8b"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    # Prepare request to Ollama
    payload = {
        "model": MODEL_NAME,
        "prompt": query,
        "stream": False  # Set to True for real-time streaming
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        # Extract the generated response (assuming it's a single string)
        generated_text = result.get("response", "")

        return jsonify({"response": generated_text})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
