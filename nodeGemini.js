import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({
  apiKey: "AIzaSyBFbbZBAr6dcnDqJUkfqsnH3oINNPKnX-w",
});

async function main() {
  const response = await ai.models.generateContent({
  model: "gemini-2.5-flash",
  contents: `Extract ingredients and common allergens from the text below.
Do not invent items.

Return JSON only in this format:
{
  "ingredients": [],
  "allergens": [],
  "warnings": []
}

Text:
[
  {
    type: 'paragraph',
    snippet: 'The Pintola Peanut Butter Creamy Roast contains roasted peanuts, sugar, a stabilizer (INS471), and iodised salt. Its primary allergen is peanuts.',
    snippet_links: [ [Object] ],
    reference_indexes: [ 1, 0 ]
  },
  { type: 'heading', snippet: 'Ingredients' },
  {
    type: 'paragraph',
    snippet: 'According to product information from sources like 1mg and Amazon.in, the ingredients for the Pintola Classic Peanut Butter Creamy are:',
    snippet_links: [ [Object], [Object] ],
    reference_indexes: [ 0, 3 ]
  },
  {
    type: 'list',
    list: [ [Object], [Object], [Object], [Object] ],
    reference_indexes: [ 0, 3, 1 ]
  },
  {
    type: 'paragraph',
    snippet: 'Note: The "All Natural" and "Organic Unsweetened" versions of Pintola peanut butter contain only 100% roasted peanuts and no other ingredients. The "Creamy Roast" mentioned in the query appears to be the "Classic" sweetened version based on the ingredient lists found.',
    reference_indexes: [ 6, 2, 7 ]
  },
  { type: 'heading', snippet: 'Allergen Information' },
  {
    type: 'paragraph',
    snippet: 'The main allergen in the product is peanuts.',
    reference_indexes: [ 4, 8 ]
  },
  {
    type: 'list',
    list: [ [Object], [Object], [Object] ],
    reference_indexes: [ 1, 4, 0, 5, 9 ]
  }
]`
});

console.log(response.text);

}

main();
