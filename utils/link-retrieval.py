import requests
import pandas as pd
import argparse
import validators
import json
import re
import os


#Setup command line arguments
parser = argparse.ArgumentParser(description="Get URL's from API and store them locally")
parser.add_argument("-e", "--endpoint", help='Target URL On Server', required=False, default="https://waldo.vision/api/trpc/urls", type=str)
parser.add_argument("-k", "--key", help='API Key', required=False, default=(os.environ.get("WALDO_API_KEY")), type=str)
parser.add_argument("-o", "--output", help='Folder To Store Output', required=True, type=str)

args = vars(parser.parse_args())

#regex function to check for valid youtube url, this does not need to be complex because were basically just checking for a video and domain
def is_valid_youtube_url(url):
    regex = r'https?://(?:www\.)?(?:youtube|youtu|youtube-nocookie|music\.youtube|gaming\.youtube|studio\.youtube|content\.googleapis|googlevideo)\.(?:com|be)/(?:watch\?v=|embed/|v/|.+\?v=)?([^\&=%\?]+)'
    return bool(re.match(regex, url))
    
if __name__ == "__main__":
    # Define the API endpoint and parameters
    endpoint = args['endpoint']
    params = {
        "auth_token": args['key'],
        "requirements": {
            "minReviews": 25,
            "rating": 90,
        }
    }
    params_json = json.dumps(params)

    # Make the API request and retrieve the data
    response = requests.get(endpoint, params=params_json)
    data = response.json()

    # Convert the data to a pandas DataFrame, this feels VERY messy, if you find a better solution, please fix.
    # Were expecting data that looks like this: [{"page": 0, "totalPages": 100, "totalGameplay": 999999, "gameplay": [{"uuid": "38r8hnf80ew-35uhehrfnjea-wryhghjvsdk", "ytUrl": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}]},{"page": 0, "totalPages": 100, "totalGameplay": 999999, "gameplay": [{"uuid": "38r8hnf80ew-35uhehrfnjea-wryh32ghjvsdk", "ytUrl": "https://www.youtube.com/watch?v=dQw24w9WgXcQ"}]}]
    response_dataframe = pd.DataFrame(data)
    response_list = response_dataframe['gameplay'].tolist()
    response_dataframe = pd.DataFrame(columns=['uuid','url'])
    for obj in response_list:
      response_dataframe = pd.concat([response_dataframe, pd.DataFrame(obj[0], index=[0])], ignore_index=True) 

    # Filter out duplicate links
    response_dataframe.drop_duplicates(subset=["url"], inplace=True)

    # Validate the URLs
    valid_urls = []
    for url in response_dataframe["url"]:
        if (validators.url(url) and is_valid_youtube_url(url)):
            valid_urls.append(url)


    # Make output directory if it doesn't exist
    download_dir = args['output']
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Save the downloaded links to a file
    valid_urls_df = pd.DataFrame(valid_urls)
    valid_urls_df.to_csv((os.path.join(download_dir + "links.csv")), index=True, columns=["uuid","url"])