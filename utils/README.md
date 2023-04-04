# utils
This directory contains all the preprocessing and utility scripts used across the models repo

### generate-requirements.sh
---
After activating a conda environment this shell script can be run to export both environment.yml and requirements.txt

### link-retrieval.py
---
Get URL's from API and stores them locally
- Options
  - -e or --endpoint is the target url on the server
    - Default: `https://waldo.vision/api/trpc/urls`
    - Required: `False`
  - -k or --key is the API Key
    - If left unspecified the default key will be pulled from the environment variable `WALDO_API_KEY`
    - Required: `False`
  - -o or --output is the folder where the file will be outputed
    - Required: `True`
    - Example: `/home/usr/website/files/` the output would then be `/home/usr/website/files/links.csv`
  - -r or --requirements is the specifications for the links you want to download
    - Default: `{"minReviews": 25, "rating": 90}`
    - Example: `{"minReviews": 20, "rating": 95}`