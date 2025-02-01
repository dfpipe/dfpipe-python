

import pytest
import pandas as pd
from dfpipe.thread import BatchDownloader
from pathlib import Path
@pytest.fixture
def downloader():
    return BatchDownloader()

def test_large_scale_download(downloader):
    # Arrange
    data = {
        'url': [
            'https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png',
            'https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png',
            'https://domain-should-not-exist.com/fake_image.jpg'
        ],
        'fpath': [
            './real-tests-tmp/google_logo_1.png',
            './real-tests-tmp/google_logo_2.png',
            './real-tests-tmp/fake_image.jpg'
        ]
    }
    df = pd.DataFrame(data)

    # Act
    result_df = downloader.download_df(df)

    # Assert
    assert result_df['exists'].sum() == 2, "Download results unexpected"
