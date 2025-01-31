
class TorchBatchDivideAndConquer:
    @staticmethod
    def divide_and_conquer(conquer_func, data_loader, merger_func):
        # conquer_func: (batch) -> result
        # data_loader: torch.utils.data.DataLoader
        # merger_func: (results) -> final_result
        results = []
        for batch in data_loader:
            result = conquer_func(batch)  # Process the batch
            results.append(result)  # Store the result
        return merger_func(results)
