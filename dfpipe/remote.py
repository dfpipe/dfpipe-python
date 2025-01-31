import time

class RemoteCall:
    @staticmethod
    def auto_retry(func, max_retries=3, retry_interval=1):
        for i in range(max_retries):
            try:
                return func()
            except Exception as e:
                time.sleep(retry_interval)
        raise IOError(f'Failed to call {func} after {max_retries} retries')