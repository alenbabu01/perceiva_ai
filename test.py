from ultralytics import YOLO
from google import genai
import requests
import json

# -------------------------------
# CONFIG
# -------------------------------
SEARXNG_URL = "http://localhost:8080/search"
GENAI_API_KEY = ""  # fill this
MODEL_PATH = "models/best.pt"

client = genai.Client(api_key=GENAI_API_KEY)

# Load local product-recognition model once
product_model = YOLO(MODEL_PATH)


# -------------------------------
# SearXNG search helper
# -------------------------------
def search_searxng(query: str):
    params = {
        "q": query,
        "format": "json"  # JSON output
    }

    try:
        response = requests.get(SEARXNG_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print("❌ Error contacting SearXNG:", e)
        return None

    return response.json()


# -------------------------------
# Local model → product name
# -------------------------------
def get_product_name_from_model(image_path: str) -> tuple[str, float]:
    """
    Use the local YOLO classification model to get the top-1 product name.
    You said: this **is** the exact model name, so we trust it.
    Returns: (product_name, confidence)
    """
    results = product_model(image_path)[0]

    top1_idx = results.probs.top1
    product_name = results.names[top1_idx]
    confidence = float(results.probs.top1conf)

    print(f"[MODEL] Predicted product: {product_name} (conf={confidence:.3f})")
    return product_name, confidence


# -------------------------------
# LLM: aggregate label info from SearXNG results
# -------------------------------
def call_genai(data, product_name: str):
    """
    Gemini ONLY extracts the best ingredients list
    and allergen details (if any).
    Product name comes entirely from the local model.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
You are given a JSON object returned by SearXNG for a query about a packaged food product.

Product name (from a local vision model, already trusted):
- {product_name}

The JSON has a top-level "results" array. Each item may contain:
- "title"
- "content" (snippet / description)
- "url"
- other metadata

This data may be incomplete, duplicated, partially wrong, or split across multiple sources.

Your task:

1. Go through **ALL** results.
2. From ALL results, extract anything that looks like:
   - an ingredients list
   - allergen statements ("contains", "allergens", etc.)
3. Build the **single most promising and complete ingredients list** by:
   - merging compatible ingredients from multiple sources,
   - preferring detailed, label-like ingredient text,
   - removing duplicates and obvious noise.
4. Extract **allergen details** only if they are explicitly mentioned.
5. Ignore:
   - recipes,
   - blogs,
   - reviews,
   unless they clearly quote the official label ingredients.
6. Prefer official brand/manufacturer/retailer-style label text when possible.

OUTPUT RULES (VERY IMPORTANT):

- Return ONLY two lines in plain text:
  Ingredients: <final merged ingredients list or empty>
  Allergens: <final merged allergen info or empty>

- If no allergen info is found, still return the line:
  Allergens:

- Do NOT return JSON.
- Do NOT add explanations.
- Do NOT add bullet points.
- Do NOT add any extra text.

---
{json.dumps(data, ensure_ascii=False)}
---
"""
    )

    print(response.text)



# -------------------------------
# Main pipeline
# -------------------------------
def process_image(image_path: str):
    # 1. Get exact product name from local model
    product_name, conf = get_product_name_from_model(image_path)

    # 2. Build query for SearXNG using that name
    query = f"{product_name} Ingredients"
    print("[INFO] Querying SearXNG with:", query)

    data = search_searxng(query)
    if data is None:
        print("[ERROR] No data from SearXNG.")
        return

    # 3. Let Gemini clean/merge the ingredients + label info
    call_genai(data, product_name)


# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    image_path = "assets/pintola.png"  # or any test image path
    process_image(image_path)
