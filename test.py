from google import genai
from PIL import Image
import pytesseract
import cv2
import easyocr

import requests
import json

client = genai.Client(api_key="AIzaSyB7ii71GL8QRoy4bWe7XA0tNyl2BATirUE")
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"



# im = cv2.imread("cola.jpg")

# text = pytesseract.image_to_string(im)
# print(text)

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



# reader = easyocr.Reader(['en'])

# results = reader.readtext('cola.jpg')

# for detection in results:
#     print(detection[1])



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
    query = input("Enter your search keyword: ")
    data = search_searxng(query)
    call_genai(data)

