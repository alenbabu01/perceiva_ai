import OpenAI from "openai";
import * as fs from "fs";
import * as path from "path";

const client = new OpenAI({ apiKey: "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" });

async function analyzeImage() {
  // Specify your local image path here
  const imagePath = "testimage.png"; // Change this to your image path

  // Read and encode the image
  const imageBuffer = fs.readFileSync(imagePath);
  const imageData = imageBuffer.toString("base64");

  // Determine the media type based on file extension
  const ext = path.extname(imagePath).toLowerCase();
  const mediaType = ext === ".png" ? "image/png" : "image/jpeg";

  const response = await client.responses.create({
    model: "gpt-4.1-mini",
    input: [{
      role: "user",
      content: [
        {
          type: "input_text",
          text: "Is this product a Dettol disinfectant spray? Only answer Yes or No"
        },
        {
          type: "input_image",
          image_url: `data:${mediaType};base64,${imageData}`
        }
      ]
    }]
  });

  console.log(response.output_text);
}

analyzeImage();