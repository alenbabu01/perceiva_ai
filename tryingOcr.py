# pip install paddleocr paddlepaddle

from paddleocr import PaddleOCR

# Initialize OCR (English)
ocr = PaddleOCR(use_textline_orientation=True, lang='en')

# Path to your image
image_path = 'cola.jpg'

# Run OCR using the new API
results = ocr.predict(image_path)

# results is a list; usually one item per image
res = results[0]

# res is a dict with keys like:
# 'rec_texts', 'rec_scores', 'rec_boxes', etc.
all_text = " ".join(res["rec_texts"])
print(all_text)
