import torch
from PIL import Image
import requests
from dfpipe.remote import RemoteContent
from io import BytesIO



class DFDataset(torch.utils.data.Dataset):
    """
    examples:
    df.columns = ['url', 'y']
    ds = DFDataset.url_y_dataset(df, image_transform=lambda x: x.size) -> read url 
    
    df.columns = ['x', 'y']
    ds = DFDataset(df)
    """

    @staticmethod
    def image_path_y_dataset(df, image_transform=None):
        # warning: this function by default assumes the image is RBG format
        assert 'image_path' in df.columns, "df must have a 'image_path' column"
        assert 'y' in df.columns, "df must have a 'y' column"
        return DFDataset(df, {'image_path': lambda x: image_transform(Image.open(x).convert('RGB'))})

    @staticmethod
    def url_y_dataset(df, image_transform=None):
        assert 'url' in df.columns, "df must have a 'url' column"
        assert 'y' in df.columns, "df must have a 'y' column"
        return DFDataset(df, {'url': ImageUrlTransform(transform=image_transform)})

    @staticmethod
    def xy_dataset(df):
        assert 'x' in df.columns, "df must have a 'x' column"
        assert 'y' in df.columns, "df must have a 'y' column"
        if not isinstance(df.iloc[0]['y'], int):
            df['y'] = df['y'].astype(int)
            print('warning: y is not an integer, try to convert it to an integer, top 5 rows of y: ', df['y'].head())
        df['x'] = df['x'].apply(lambda x: torch.tensor(x))
        return DFDataset(df)

    def __init__(self, df, data_transform: dict={}):
        self.df = df
        self.data_transform = data_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx].to_dict()
        for key, transform in self.data_transform.items():
            item[key] = transform(item[key])
        return item

class ImageUrlTransform:
    """
    Transform a url to an image.
    """
    def __init__(self, format='RGB', transform=None):
        self.format = format
        self.transform = transform

    def __call__(self, url):
        content = RemoteContent(url).read_auto_retry()
        image = Image.open(BytesIO(content)).convert(self.format)
        if self.transform is not None:
            image = self.transform(image)
        return image




