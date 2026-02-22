import asyncio
import aiohttp
import base64
import io
import os
import random
import re
from PIL import Image
from pathlib import Path
from aiohttp import ClientConnectorError, ServerDisconnectedError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# ================= CONFIG =================
OPENAI_URL = "https://api.openai.com/v1/responses"
CROPS_DIR = r"assets/croppedImages"  # Update this to your actual image folder

BATCH_SIZE = 1
MAX_IMAGE_SIZE = 1600
JPEG_QUALITY = 100
USE_PNG = True  # Better for small text; set False to use JPEG
MAX_CONCURRENT_REQUESTS = 5
MAX_RETRIES = 3
# =========================================


PRODUCT_NAME = "Maggi"

PROMPT = f"""
Is the product shown in this image {PRODUCT_NAME}?

Answer only YES or NO.
"""


# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(path):
    with open(path, "rb") as f:
        img_bytes = f.read()

    # Infer MIME type from file extension
    ext = path.suffix.lower()
    if ext == ".png":
        mime = "image/png"
    elif ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    else:
        raise ValueError(f"Unsupported image format: {ext}")

    return img_bytes, mime



_ANSWER_RE = re.compile(r"^\s*IMG\s*(\d+)\s*:\s*(YES|NO)\s*$", re.IGNORECASE)


def _parse_yes_indices(output_text: str, batch_len: int) -> list[int]:
    yes_indices: set[int] = set()
    for line in (output_text or "").splitlines():
        match = _ANSWER_RE.match(line)
        if not match:
            continue
        idx = int(match.group(1)) - 1
        label = match.group(2).upper()
        if 0 <= idx < batch_len and label == "YES":
            yes_indices.add(idx)
    return sorted(yes_indices)

# ---------------- BATCH HELPER ----------------
def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# ---------------- GPT CALL ----------------
async def classify_batch(session, batch, stop_event, semaphore):
    async with semaphore:
        if stop_event.is_set():
            return None

        content = [{
            "type": "input_text",
            "text": PROMPT
        }]

        for p in batch:
            img_bytes, mime = preprocess_image(p)
            b64 = base64.b64encode(img_bytes).decode()
            content.append({
                "type": "input_image",
                "image_url": f"data:{mime};base64,{b64}"
            })

        payload = {
            "model": "gpt-4.1-mini",
            "text": {"verbosity": "medium"},
            "input": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        }

        headers = {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json"
        }

        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(
                    OPENAI_URL,
                    json=payload,
                    headers=headers,
                    timeout=60
                ) as r:

                    if r.status != 200:
                        err = await r.text()
                        raise RuntimeError(err)

                    data = await r.json()

                # -------- OUTPUT EXTRACTION (CORRECT) --------
                output = ""

                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for part in item.get("content", []):
                            if part.get("type") == "output_text":
                                output += part.get("text", "")

                # yes_indices = _parse_yes_indices(output, len(batch))

                # if yes_indices:
                #     stop_event.set()
                #     return [batch[i] for i in yes_indices]  # only the crop(s) that were YES

                # return None

                answer = output.strip().upper()
                if answer == "YES":
                    stop_event.set()
                    return batch  # single image
                return None




            except (ServerDisconnectedError, ClientConnectorError,
                    asyncio.TimeoutError, RuntimeError) as e:

                if attempt == MAX_RETRIES - 1:
                    print("❌ Request failed after retries:", e)
                    return None

                await asyncio.sleep(2 ** attempt + random.random())

# ---------------- CONTROLLER ----------------
async def run_batching():
    image_paths = list(Path(CROPS_DIR).glob("*.jpg"))

    if not image_paths:
        print("❌ No cropped images found")
        return

    batches = list(chunked(image_paths, BATCH_SIZE))
    print(f"🧠 {len(image_paths)} crops → {len(batches)} batches")

    stop_event = asyncio.Event()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    timeout = aiohttp.ClientTimeout(total=90)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            asyncio.create_task(
                classify_batch(session, b, stop_event, semaphore)
            )
            for b in batches
        ]

        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                if result:
                    for t in tasks:
                        t.cancel()

                    print("🎯 TARGET PRODUCT FOUND — STOPPING")
                    print("📍 Location(s):")
                    for img in result:
                        print("   →", img)

                    return

            except asyncio.CancelledError:
                pass

    print("❌ TARGET PRODUCT NOT FOUND")

if __name__ == "__main__":
    asyncio.run(run_batching())