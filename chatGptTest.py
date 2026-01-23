from openai import OpenAI
import base64
import sys

client = OpenAI(api_key="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")

# Specify your local image path here
image_path = "testimage.png"  # Change this to your image path

# Read and encode the image
with open(image_path, "rb") as image_file:
    image_data = base64.standard_b64encode(image_file.read()).decode("utf-8")

# Determine the media type based on file extension
media_type = "image/png" if image_path.endswith(".png") else "image/jpeg"

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Is this product a Dettol disinfectant spray? Only answer Yes or No"},
            {
                "type": "input_image",
                "image_url": f"data:{media_type};base64,{image_data}",
            },
        ],
    }],
)

print(response.output_text)