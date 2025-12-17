import requests
import os

url = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"
output_path = "fake_or_real_news.csv"

if not os.path.exists(output_path):
    print(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading: {e}")
else:
    print("Dataset already exists.")
