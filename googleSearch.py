import requests

def google_search(query, api_key, cx):
    """
    Perform a Google search using the Custom Search JSON API.
    
    Parameters:
        query (str): The search query string.
        api_key (str): Your Google API key.
        cx (str): Your Programmable Search Engine ID.
    
    Returns:
        dict: JSON response containing search results.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,       # search query
        "key": api_key,   # your API key
        "cx": cx          # your search engine ID
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

# Example usage:
if __name__ == "__main__":
    API_KEY = "heh"  # replace with your API key
    CX = "heh"  # replace with your search engine ID
    query = "Python programming tutorials"
    
    results = google_search(query, API_KEY, CX)
    print(results)
