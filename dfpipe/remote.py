import time
import requests

class RemoteCall:
    @staticmethod
    def auto_retry(func, max_retries=3, retry_interval=0.01):
        for i in range(max_retries):
            try:
                return func()
            except Exception as e:
                time.sleep(retry_interval)
        raise IOError(f'Failed to call {func} after {max_retries} retries')

class RemoteContent:
    def __init__(self, url):
        self.url = url

    def read_auto_retry(self):
        return RemoteCall.auto_retry(self.read_bytes)

    def read_bytes(self):
        response = requests.get(self.url)
        if response.status_code != 200:
            raise IOError(f'Failed to download {self.url}')
        return response.content