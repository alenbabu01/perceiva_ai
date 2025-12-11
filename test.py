from ultralytics import YOLO
import requests
import json
from serpapi import GoogleSearch


# -------------------------------
# CONFIG
# -------------------------------
SERPAPI_KEY = ""
LOCAL_LLM_URL = "http://127.0.0.1:5000/generate"
MODEL_PATH = "models/best.pt"

# Load YOLO model
product_model = YOLO(MODEL_PATH)


# -------------------------------
# SerpApi search
# -------------------------------
def search_serpapi(query: str):
    params = {
        "engine": "google_ai_mode",
        "q": query,
        "api_key": SERPAPI_KEY
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        print("Results from the ai :",results["text_blocks"])
        return results["text_blocks"]
    except Exception as e:
        print("❌ Error contacting SerpApi:", e)
        return None


# -------------------------------
# Get product name from model
# -------------------------------
def get_product_name_from_model(image_path: str) -> tuple[str, float]:
    results = product_model(image_path)[0]

    top1_idx = results.probs.top1
    product_name = results.names[top1_idx]
    confidence = float(results.probs.top1conf)

    print(f"[MODEL] Predicted product: {product_name} (conf={confidence:.3f})")
    return product_name, confidence


# -------------------------------
# Local LLM helper
# -------------------------------
def generate_with_local_llm(prompt: str) -> str:
    try:
        resp = requests.post(
            LOCAL_LLM_URL,
            json={"query": prompt},
            timeout=120
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.RequestException as e:
        print("❌ Error contacting local LLM:", e)
        return ""


# -------------------------------
# Ingredient extraction prompt
# -------------------------------
def call_local_llm(data, product_name: str):
    prompt = f"""
You are a text-processing function, NOT a chat assistant.

Your job:
- Extract ingredients + allergen info for {product_name}
- Output exactly:

Ingredients: ...
Allergens: ...

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
    # 1. Recognize product
    product_name, conf = get_product_name_from_model(image_path)

    # 2. Query SerpApi instead of SearXNG
    query = f"{product_name} ingredients and contain any allergens?"
    print("[INFO] Querying SerpApi with:", query)

    data = search_serpapi(query)
    if data is None:
        print("[ERROR] No data from SerpApi.")
        return

    # 3. Extract ingredients using local LLM
    call_local_llm(data, product_name)


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    image_path = "assets/image.png"
    process_image(image_path)
