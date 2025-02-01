
import pytest
import pandas as pd
from pathlib import Path
from dfpipe.torch.data import DFDataset
import time
import torch

def _generate_url_y_df():
    data = {
        'url': [
            'https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png',
            'https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png',
        ],
        'y': [
            1,
            1,
        ]
    }
    df = pd.DataFrame(data)
    return df

def url_y_dataset_load():
    # Testing a simple image downloader, roughly 10 url per second
    n = 10
    df = pd.DataFrame({'url': ['https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png'] * n, 'y': [1] * n})


    ds = DFDataset.url_y_dataset(df, image_transform=lambda x: x.size)
    for item in ds:
        assert item['url'] == (272, 92)

def url_y_dataset_batch_loader():
    n = 100
    df = pd.DataFrame({'url': ['https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png'] * n, 'y': [1] * n})
    ds = DFDataset.url_y_dataset(df, image_transform=lambda x: torch.tensor(x.size))
    
    dataloader = torch.utils.data.DataLoader(ds, batch_size=10, num_workers=8)
    for batch in dataloader:
        print(batch)


def test_url_y_dataset_sequential_speed(benchmark):
    result = benchmark(url_y_dataset_load)
    assert result is None

def test_url_y_dataset_parallel_speed(benchmark):
    result = benchmark(url_y_dataset_batch_loader)
    assert result is None