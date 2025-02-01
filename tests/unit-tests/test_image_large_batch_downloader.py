import pytest
from unittest.mock import patch, MagicMock
from dfpipe.image import ImageLargeBatchDownloader

@pytest.fixture
def downloader():
    return ImageLargeBatchDownloader()

@patch('dfpipe.image.requests.get')
def test_download_image_success(mock_get, downloader):
    # Arrange
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = b'fake_image_data'
    url = 'http://example.com/fake_image.jpg'
    image_path = '/tmp/fake_image.jpg'

    # Act
    downloader.download_image(url, image_path)

    # Assert
    assert downloader.failed_urls == []
    mock_get.assert_called_once_with(url)

@patch('dfpipe.image.requests.get')
def test_download_image_failure(mock_get, downloader):
    # Arrange
    mock_get.side_effect = Exception('Network error')
    url = 'http://example.com/fake_image.jpg'
    image_path = '/tmp/fake_image.jpg'

    # Act
    downloader.download_image(url, image_path)

    # Assert
    assert url in downloader.failed_urls

@patch('dfpipe.image.requests.get')
def test_download_images(mock_get, downloader):
    # Arrange
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = b'fake_image_data'
    url_list = ['http://example.com/fake_image1.jpg', 'http://example.com/fake_image2.jpg']
    image_path_list = ['/tmp/fake_image1.jpg', '/tmp/fake_image2.jpg']

    # Act
    downloader.download_images(url_list, image_path_list)

    # Assert
    assert downloader.failed_urls == []
    assert mock_get.call_count == len(url_list)
