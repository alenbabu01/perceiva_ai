from flask import Flask, request, jsonify
import requests
import json
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --------- CONFIG ----------
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:8b"
DEFAULT_TEMPERATURE = 0
DEFAULT_TIMEOUT = 120
# --------------------------

SYSTEM_INSTRUCTION = (
    "You are a helpful, safety-conscious food-safety assistant. "
    "Produce a short, friendly natural-language message the user can read. "
    "Be concise, prioritize allergies and intolerances, and provide clear next steps."
)


def call_ollama(prompt: str, temperature: float = DEFAULT_TEMPERATURE, timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Send the assembled prompt to Ollama and return the model text.
    Tolerant to common Ollama response shapes.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature
    }
    try:
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.RequestException as e:
        logging.exception("Error contacting Ollama")
        raise RuntimeError(str(e))

    # Try common fields
    model_text = result.get("response")
    if not model_text:
        outputs = result.get("outputs")
        if isinstance(outputs, list) and outputs:
            model_text = outputs[0].get("text") or outputs[0].get("content") or outputs[0].get("response")
    if not model_text:
        # fallback: return stringified whole response
        model_text = json.dumps(result, ensure_ascii=False)

    return str(model_text).strip()


@app.route("/generate", methods=["POST"])
def generate():
    """
    Expected JSON body:
    {
      "product_name": "string",
      "user_profile": { ... },     # same shape as sample_user_data["user_profile"]
      "search_data": "<string or object containing ingredient/label text>"
    }

    Returns:
      {"response": "<natural-language advice>"}
    """
    req = request.get_json(silent=True)
    if not req:
        return jsonify({"error": "Missing JSON body"}), 400

    product_name = req.get("product_name")
    user_profile = req.get("user_profile")
    search_data = req.get("search_data", "")

    # Basic validation
    if not product_name:
        return jsonify({"error": "Missing required field: product_name"}), 400
    if not user_profile:
        return jsonify({"error": "Missing required field: user_profile"}), 400

    # Prepare user_profile and search_data blobs
    try:
        user_profile_blob = json.dumps(user_profile, ensure_ascii=False, indent=2)
    except Exception:
        user_profile_blob = str(user_profile)

    if isinstance(search_data, (dict, list)):
        try:
            search_blob = json.dumps(search_data, ensure_ascii=False, indent=2)
        except Exception:
            search_blob = str(search_data)
    else:
        search_blob = str(search_data)

    # Build the prompt
    prompt = f"""
SYSTEM:
{SYSTEM_INSTRUCTION}

CONTEXT:
- product_name: {product_name}
- user_profile: {user_profile_blob}

Search/label data (may include ingredient lists, product descriptions, or label text):
{search_blob}

TASK (follow steps in order; return only the natural-language advice):
1) On one short line each, state the extracted Ingredients and explicit Allergens you find:
   Ingredients: ...
   Allergens: ...
   If none found, write 'none found' for that line.

2) Based only on the user's profile above (prioritize allergies/intolerances first, then health conditions, then diet/restrictions),
   write a short (2-4 sentence) assessment stating whether the product is likely safe, possibly unsafe, or unsafe for this user.

3) Provide 2-4 concise, actionable next steps (each on its own line). Examples:
   - "Check label for: milk, whey, casein, peanut, tree nuts, gluten, sulfites, sodium"
   - "If allergic to X, avoid or choose an alternative Y"
   - "If diabetic, note sugar content or avoid if high"

4) If ingredients cannot be found or the data is ambiguous, say so explicitly and list exact label keywords the user should look for.

Tone: friendly, non-alarming, clear, concise. Do NOT output JSON or debugging metadata — only readable advice.

--- End prompt ---
"""

    # Call the model
    try:
        model_text = call_ollama(prompt, temperature=DEFAULT_TEMPERATURE, timeout=DEFAULT_TIMEOUT)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    # Return natural-language response
    return jsonify({"response": model_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
