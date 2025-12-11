from flask import Flask, request, jsonify
import requests
import json

app = Flask(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:12b"

SYSTEM_INSTRUCTION = (
    "You are a strict text-processing function. Your only job is to scan the provided JSON "
    "and return two lines exactly: 'Ingredients: ...' and 'Allergens: ...'. Do not add anything else."
)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json or {}
    user_query = data.get("query", "")
    if not user_query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    prompt = (
        f"SYSTEM:\n{SYSTEM_INSTRUCTION}\n\n"
        f"USER:\n{user_query}\n\n"
        f"---\n"
        "Return EXACTLY two lines:\n"
        "Ingredients: <text or empty>\n"
        "Allergens: <text or empty>\n"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.0
    }

    try:
        # shorter timeout to reduce wait time
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        # Return model's text directly (common field names)
        model_text = result.get("response")
        if model_text is None:
            # fallback to other possible field shapes
            if isinstance(result.get("outputs"), list) and result["outputs"]:
                model_text = result["outputs"][0].get("text")
            else:
                model_text = ""

        return jsonify({"response": model_text})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
