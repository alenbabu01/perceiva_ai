import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({
  apiKey: "AIzaSyCfuco7hN5dqcTNmH0Y0TGN70o1qGGpa74",
});

async function main() {
  const response = await ai.models.generateContent({
  model: "gemini-2.5-flash",
  contents: `
You are an intent classification engine for a voice assistant used by a blind person.

User said:
"What am I holding right now?"

Available backend features:
1. product_identification – identify product using camera
2. medical_compatibility_check – check if food or medicine is safe
3. ocr_reading – read text from labels
4. volunteer_call – connect user to a volunteer
5. general_assistance – general help or questions

Rules:
- Choose ONLY ONE feature
- Respond ONLY in JSON
- Do NOT explain anything

Response format:
{
  "intent": "<feature_name>",
  "confidence": <number between 0 and 1>,
  "reason": "<short reason>"
}
`
});

console.log(response.text);

}

main();
