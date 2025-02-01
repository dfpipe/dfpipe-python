import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd

class BatchDownloader:
    def __init__(self):
        self.max_size = 4096
        self.max_workers = 20
        self.failed_urls = []

    def download_one(self, url, file_path):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            self.failed_urls.append(url)

    def download_df(self, df: pd.DataFrame):
        assert 'url' in df.columns, "df must have a 'url' column"
        assert 'fpath' in df.columns, "df must have a 'fpath' column"

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for _, row in df.iterrows():
                futures.append(executor.submit(self.download_one, row['url'], row['fpath']))

            for future in tqdm(futures):
                future.result()  # Wait for all futures to complete

        df['exists'] = df['fpath'].apply(lambda x: Path(x).exists())
        return df