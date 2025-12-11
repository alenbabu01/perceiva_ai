from serpapi import GoogleSearch

params = {
  "engine": "google_ai_overview",
  "q": "Bingo_Tomato_Potato_Chips ingredients and contain any allergens?",
  "api_key": "5be268df7d4b5bdbc09b2637987f081edcb39ae19853bf4dcacb376e77bc7761"
}

search = GoogleSearch(params)
results = search.get_dict()
ai_overview = results["ai_overview"]
print(ai_overview)