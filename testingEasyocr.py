import easyocr

# Step 1: Initialize the reader
# Add the languages you want to recognize, e.g. English ('en')
reader = easyocr.Reader(['en'])

# Step 2: Run OCR on your image
results = reader.readtext('cola.jpg')

# Step 3: Print the recognized text
for detection in results:
    print(detection[1])
