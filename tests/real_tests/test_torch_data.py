
import pytest
import pandas as pd
from pathlib import Path
from dfpipe.torch.data import DFDataset

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

def test_url_y_dataset():
    # Testing a simple image downloader
    df = _generate_url_y_df()

    ds = DFDataset.url_y_dataset(df, image_transform=lambda x: x.size)
    assert len(ds) == 2

    item = ds[0]
    assert item['url'] == (272, 92)
    assert item['y'] == 1

    item = ds[1]
    assert item['url'] == (272, 92)
    assert item['y'] == 1


