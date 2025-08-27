import os
import requests
from dotenv import load_dotenv

load_dotenv()

DATA_API_URL = os.getenv('DATA_API_URL')

def data_fetch_and_save():
   
    response = requests.get(url=DATA_API_URL)

    if response.status_code == 200:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "data")
        os.makedirs(data_dir, exist_ok=True)

        filename = f"{project_root}/data/finviz_raw.csv"

        with open(filename, "wb") as f:
            f.write(response.content)

        print(f"Data fetched and saved to {filename}")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")


data_fetch_and_save()