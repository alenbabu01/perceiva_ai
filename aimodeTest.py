# import requests

# url = "https://www.searchapi.io/api/v1/search"
# params = {
#     "engine": "google_ai_mode",
#     "url": "https://images.picxy.com/cache/2020/10/19/408304a9a62894f43dbf06dedf065cf0.webp",
#     "q": "Are you able to find wagh bakri tea powder in this shelf ? Just tell Yes or No",
#     "api_key": "hbiM3xX6cGw2JVBBfRK8p6RN"
# }

# response = requests.get(url, params=params)
# result = response.json()
# print(result)



from serpapi import GoogleSearch

params = {
    "engine": "google_ai_mode",
    "url": "https://i.ibb.co/12qRDsk/tower.png",   # public image URL
    "q": "What is this image ?",
    "api_key": "5be268df7d4b5bdbc09b2637987f081edcb39ae19853bf4dcacb376e77bc7761"
}

search = GoogleSearch(params)
results = search.get_dict()

text_blocks = results.get("text_blocks", [])
markdown = results.get("markdown", "")
references = results.get("reference_links", [])

print(text_blocks)

