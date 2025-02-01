import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class ImageLargeBatchDownloader:
    def __init__(self):
        self.failed_urls = []

    def download_image(self, url, image_path):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                Path(image_path).parent.mkdir(parents=True, exist_ok=True)
                with open(image_path, 'wb') as f:
                    f.write(response.content)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            self.failed_urls.append(url)

    def download_images(self, url_list, image_path_list):
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for url, image_path in zip(url_list, image_path_list):
                futures.append(executor.submit(self.download_image, url, image_path))

            for future in tqdm(futures):
                future.result()  # Wait for all futures to complete