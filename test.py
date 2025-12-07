from google import genai
from PIL import Image
import pytesseract
import cv2
import easyocr

import requests
import json

client = genai.Client(api_key="")
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"


SEARXNG_URL = "http://localhost:8888/search"


def search_searxng(query: str):
    params = {
        "q": query,
        "format": "json"   # very important for JSON output
    }

    try:
        response = requests.get(SEARXNG_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print("❌ Error contacting SearXNG:", e)
        return

    data = response.json()

    # Pretty print JSON
    return data

reader = easyocr.Reader(['en'])  # create once, reuse


def extract_product_name_from_ocr(image_path: str) -> tuple[str, str]:

    
    # results = reader.readtext(image_path) 

    # ocr_text = " ".join([r[1] for r in results])

    results = cv2.imread(image_path)
    results.view()

    ocr_text = pytesseract.image_to_string(results)
    print()


    print("This is the OCR text",ocr_text)

    # Ask LLM to extract product name
    resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=f"""
You are given noisy OCR text captured from the packaging of a **single packaged food or drink product**.

You must identify the **full product name** (brand + product + variant) even if the OCR text is incomplete.

OCR text:
---
{ocr_text}
---

Your task:

1. Infer the most likely **exact product name as printed on the front of the pack**.
   - Use direct matches from the OCR text when available.
   - But also use **cues and associations** when the brand or variant is missing, such as:
     - flavor names (e.g. "Magic Masala", "Masala Twist", "Cold Coffee")
     - slogans or taglines specific to known brands
     - abbreviations or stylized spellings
     - references to shape/packaging (e.g. "Classic Salted" hints at Lay's Classic Salted)
     - popular product series names (e.g. "Dairy Milk Silk", "Oreo Double Stuf", etc.)
     - well-known brand color schemes (if mentioned by OCR, e.g. "red label", "blue classic", etc.)
   - If OCR text is messy, combine compatible fragments to form the most likely real product name.

2. If you have multiple possibilities, pick the **most widely recognized / most plausible retail product**.

3. Return format:
   - If you are reasonably confident: **return ONLY the product name** (one line, no quotes, no explanation).
   - If NOT reasonably confident: return exactly:
     PRODUCT_NAME_NOT_FOUND

Examples of reasoning you are allowed to apply internally (do NOT output the reasoning):
- "Magic Masala" strongly implies "Lay's Indian Magic Masala".
- "Silk Oreo" strongly implies "Cadbury Dairy Milk Silk Oreo".
- "Zero Sugar Cola" strongly implies "Coca-Cola Zero Sugar" if context suggests cola.
- "Crunchy Masala Noodles" implies "Maggi Masala" or "Yippee Magic Masala" depending on other cues.

IMPORTANT:
- DO NOT return generic category words like "Potato Chips", "Cold Drink", "Noodles".
- DO NOT return company name or factory address.
- DO NOT add quotes, punctuation, or commentary. Just the product name.
"""
)


    product_name = resp.text.strip()
    print("This is the identified product name ", product_name)
    return product_name, ocr_text


def process_image(image_path: str):
    # 1. OCR + product name extraction
    product_name, ocr_text = extract_product_name_from_ocr(image_path)

    if product_name == "PRODUCT_NAME_NOT_FOUND":
        # Fallback: use raw OCR text as query
        print("[INFO] Product name not confidently identified, falling back to OCR text for search.")
        query = ocr_text + "Ingredients"
    else:
        print(f"[INFO] Detected product name: {product_name}")
        query = product_name + "Ingredients"

    # 2. Search SearXNG
    data = search_searxng(query)   # must RETURN the SearXNG JSON dict

    # 3. Let your existing LLM prompt merge + filter ingredients/allergens/etc.
    call_genai(data)   # uses the big "go through all results" prompt we wrote


def call_genai(data):

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
You are given a JSON object returned by SearXNG for a query about a packaged food product.

The JSON has a top-level "results" array. Each item may contain fields like:
- "title"
- "content" (snippet / description)
- "url"
- other metadata

The information may be:
- incomplete,
- partially wrong,
- duplicated across sites,
- or split across multiple results.

Your task:

1. Look at **ALL** results, not just one.
   - From every result, extract any text that looks like it contains **ingredients / allergens / label info**.
   - Pay attention to parts starting with or containing keywords like:
     "Ingredients:", "Contains", "Allergens", "May contain", "Suitable for", "Not suitable for", etc.

2. From ALL the candidate snippets collected across the results, infer the **most correct and complete ingredients list**:
   - Prefer ingredients that:
     - appear consistently across multiple sources,
     - come from more "official" looking pages (brand / manufacturer / FSSAI / retailer product page),
     - look like detailed label text rather than short, vague descriptions.
   - If two sources conflict:
     - Prefer the one that is more detailed and complete.
     - Prefer more recent-looking or more official-sounding label text.
   - It is allowed to merge pieces from multiple sites if they clearly refer to the same product and are compatible.
   - Remove obvious duplicates, re-order if needed to make a clean list.

3. In addition to ingredients, also aggregate as MUCH label information as possible from all results:
   - allergens: explicit allergen statements (e.g. "contains milk, soy, wheat")
   - may_contain: "may contain" / "processed in facility" style warnings
   - suitability: dietary suitability (e.g. vegetarian / vegan / egg / non-veg / gluten-free)
   - warnings: any safety / health warnings (e.g. "not suitable for children", "contains phenylalanine")
   - additives_info: info about additives / E-numbers / preservatives / colours / flavours
   - other_label_info: any other important label text (e.g. "no added sugar", "contains artificial sweeteners")
   - source_url: one or more URLs (comma-separated) of the most important sources you relied on

4. Be robust to noise:
   - Ignore recipes, blogs and reviews **unless** they clearly quote the official ingredients / label text.
   - Clean spacing, remove obvious marketing fluff, but KEEP all useful safety & allergen information.
   - If a snippet obviously describes a different product, ignore it.

5. Output format (VERY IMPORTANT):
   - Return a SINGLE JSON object with these keys:
     {{
       "ingredients": "<string or empty if unknown>",
       "allergens": "<string or empty if unknown>",
       "may_contain": "<string or empty if unknown>",
       "suitability": "<string or empty if unknown>",
       "warnings": "<string or empty if unknown>",
       "additives_info": "<string or empty if unknown>",
       "other_label_info": "<string or empty if unknown>",
       "source_url": "<string or empty if unknown>"
     }}
   - Each value must be a plain string (no lists, no nested JSON).
   - If you list multiple URLs in source_url, separate them by commas in a single string.
   - Do NOT wrap this object in markdown, do NOT add any explanation or extra text. Output ONLY the JSON.

If you are not reasonably sure you found any valid product label information, return exactly this JSON:
{{
  "ingredients": "",
  "allergens": "",
  "may_contain": "",
  "suitability": "",
  "warnings": "",
  "additives_info": "",
  "other_label_info": "",
  "source_url": "",
  "error": "PRODUCT_INFO_NOT_FOUND"
}}

Here is the JSON from SearXNG:
---
{json.dumps(data, ensure_ascii=False)}
---
"""
    )

    print(response.text)



if __name__ == "__main__":
    image_path = "maggi.png"   # or input() / CLI arg
    process_image(image_path)
