from ultralytics import YOLO
import requests
import json

# -------------------------------
# CONFIG
# -------------------------------
SEARXNG_URL = "http://localhost:8080/search"
LOCAL_LLM_URL = "http://127.0.0.1:5000/generate"
MODEL_PATH = "models/best.pt"

# Load local product-recognition model once
product_model = YOLO(MODEL_PATH)


# -------------------------------
# SearXNG search helper
# -------------------------------
def search_searxng(query: str):
    params = {
        "q": query,
        "format": "json"
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
    results = product_model(image_path)[0]

    top1_idx = results.probs.top1
    product_name = results.names[top1_idx]
    confidence = float(results.probs.top1conf)

    print(f"[MODEL] Predicted product: {product_name} (conf={confidence:.3f})")
    return product_name, confidence


# -------------------------------
# Local LLM helper (MATCHES YOUR FLASK SERVER)
# -------------------------------
def generate_with_local_llm(prompt: str) -> str:
    try:
        resp = requests.post(
            LOCAL_LLM_URL,
            json={"query": prompt},
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()

        return data.get("response", "").strip()

    except requests.RequestException as e:
        print("❌ Error contacting local LLM:", e)
        return ""


# -------------------------------
# LLM: extract ingredients + allergens ONLY
# -------------------------------


def call_local_llm(data, product_name: str):
    prompt = f"""
You are a text-processing function, NOT a chat assistant.

Your ONLY job:
- Read the JSON data.
- Extract the best possible INGREDIENTS LIST and ALLERGEN INFO for this product:
  {product_name}

IMPORTANT BEHAVIOUR:

1. Look through ALL results in the JSON.
2. From anywhere in the JSON, collect text that looks like:
   - ingredients list
   - allergen statements ("contains", "allergens", "may contain", etc.)
3. Merge and clean this into:
   - One single ingredients line.
   - One single allergens line.
4. You may merge ingredients from multiple sources.
5. Ignore recipes/blogs unless they clearly quote the product label.

NOW THE MOST IMPORTANT PART:

You MUST respond in EXACTLY this format, with EXACTLY two lines:

Ingredients: <final merged ingredients list or empty>
Allergens: <final merged allergen info or empty>

Rules:
- No extra spaces before "Ingredients:" or "Allergens:".
- If there are no allergens, still output: Allergens:
- Do NOT output anything else. No explanations, no markdown, no headings, no lists.
- Your entire reply must match this pattern:
  Ingredients: ...
  Allergens: ...

Here is the JSON data (do NOT repeat or summarize it, just use it silently):

<DATA>
{json.dumps(data, ensure_ascii=False)}
</DATA>
"""

    output = generate_with_local_llm(prompt)
    print("\n✅ FINAL OUTPUT FROM LOCAL LLM:\n")
    print(output)


# -------------------------------
# Main pipeline
# -------------------------------
def process_image(image_path: str):
    # 1. Get exact product name from local model
    product_name, conf = get_product_name_from_model(image_path)

    # 2. Build query for SearXNG
    query = f"{product_name} Ingredients"
    print("[INFO] Querying SearXNG with:", query)

    data = search_searxng(query)
    if data is None:
        print("[ERROR] No data from SearXNG.")
        return

    # 3. Use LOCAL LLM (not Gemini)
    call_local_llm(data, product_name)


# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    image_path = "assets/image.png"
    process_image(image_path)
