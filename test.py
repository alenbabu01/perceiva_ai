from ultralytics import YOLO
import requests
import json
from serpapi import GoogleSearch


# -------------------------------
# CONFIG
# -------------------------------
SERPAPI_KEY = "5be268df7d4b5bdbc09b2637987f081edcb39ae19853bf4dcacb376e77bc7761"
LOCAL_LLM_URL = "http://127.0.0.1:5000/generate"
MODEL_PATH = "models/best.pt"

sample_user_data = {
  "user_profile": {
    "allergies": [
      "peanuts",
      "milk",
      "sulfites"
    ],

    "intolerances": [
      "lactose",
      "gluten"
    ],

    "health_conditions": [
      "diabetes",
      "hypertension"
    ],

    "diet_type": "vegetarian",

    "food_restrictions": [
      "low_sodium",
      "no_artificial_sweeteners",
      "caffeine_free"
    ],

    "medication_food_interactions": [
      "avoid_grapefruit",
      "avoid_high_vitamin_k_foods"
    ],

    "age_group": "adult",

    "preferences_optional": {
      "calorie_limit_per_day": 1800,
      "avoid_food_textures": ["greasy"],
      "preferred_macros": {
        "protein": "high",
        "carbs": "medium",
        "fat": "low"
      }
    }
  }
}


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
def generate_with_local_llm(product_name: str, search_data, user_profile: dict):
    payload = {
        "product_name": product_name,
        "user_profile": user_profile,
        "search_data": search_data
    }

    try:
        resp = requests.post(LOCAL_LLM_URL, json=payload, timeout=120)
    except requests.RequestException as e:
        print("❌ Error contacting local LLM:", e)
        return ""

    if resp.status_code != 200:
        print(f"❌ Local LLM returned: {resp.status_code}")
        print("Response body:", resp.text)
        return ""

    try:
        data = resp.json()
    except:
        print("❌ Non-JSON response from LLM:", resp.text)
        return ""

    return data.get("response", "")



# -------------------------------
# Ingredient extraction prompt
# -------------------------------
# -------------------------------
# Ingredient extraction + safety advice (REPLACEMENT)
# -------------------------------
def call_local_llm(data, product_name: str):
    """
    Uses the local LLM to:
      1) extract ingredients + obvious allergens from the `data` blob returned by SerpApi,
      2) compare them against the global `sample_user_data` profile,
      3) produce a natural-language safety assessment and actionable advice for the user.

    The output is printed and also returned as a string.
    """

    # Prepare compact JSON snippets for optional logging/debugging (safe to send full data to LLM)
    try:
        search_blob = json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        search_blob = str(data)

    try:
        user_profile_blob = json.dumps(sample_user_data["user_profile"], ensure_ascii=False, indent=2)
    except Exception:
        user_profile_blob = str(sample_user_data.get("user_profile", {}))

    # Call the local LLM endpoint with the structured payload (server expects product_name, user_profile, search_data)
    output = generate_with_local_llm(
        product_name=product_name,
        search_data=data,
        user_profile=sample_user_data["user_profile"]
    )

    if not output:
        output = (
            "Sorry, I couldn't contact the local LLM to extract ingredients and assess safety. "
            "Please check the product label for ingredients and allergens (e.g., milk, peanuts, wheat/gluten, "
            "sulfites). If you want, paste the ingredient list here and I can evaluate it for you."
        )

    # Optional: small debug print of what we sent (comment out in production)
    print(f"\n[DEBUG] Sent to LLM - product_name: {product_name}")
    print(f"[DEBUG] Sent to LLM - user_profile: {user_profile_blob}")
    print(f"[DEBUG] Sent to LLM - search_data (truncated): {search_blob[:1000]}")  # avoid huge prints

    # Print and return the LLM's advice
    print("\n✅ FINAL ADVICE FROM LOCAL LLM:\n")
    print(output)
    return output


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
