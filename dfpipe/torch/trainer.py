# this package requires torch and torchvision
import torch
from torch import nn
import time
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from dfpipe.torch.ext import TorchBatchDivideAndConquer
from tqdm import tqdm

class TorchModelTrainer:
    def fit(self, model_parameters, forward_function, ds, epoch_num=10, optimizer_name='adamw', device='cuda:0', loss_name='cross_entropy', epoch_end_callback=None) -> dict:
        # return a dictionary of metrics
        
        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(model_parameters, lr=1e-3)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model_parameters, lr=1e-3)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model_parameters, lr=1e-3)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        print('optimizer: ', optimizer)

        if loss_name == 'cross_entropy':
            loss_function = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

        dataloader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=4)
        metrics = {
            'epoch': [],
            'loss': {},
            'batch_accuracy': {},
        }
        for epoch in range(epoch_num):
            metrics['epoch'].append(epoch)
            for batch in tqdm(dataloader, desc=f'training for epoch {epoch} '):
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                optimizer.zero_grad()
                y_pred = forward_function(x)
                if loss_name == 'cross_entropy':
                    loss = loss_function(y_pred, y)
                    batch_accuracy = (y_pred.argmax(dim=1) == y).float().mean().item()
                    metrics['batch_accuracy'].setdefault(epoch, []).append(batch_accuracy)
                    metrics['loss'].setdefault(epoch, []).append(loss.item())
                else:
                    raise ValueError(f"Unsupported loss: {loss_name}")
                loss.backward()
                optimizer.step()
            if epoch_end_callback is not None:
                try:
                    epoch_end_callback(metrics=metrics)
                except Exception as e:
                    print(f'warning: epoch_end_callback failed at epoch {epoch}: {e}')
                    print(e)
                    print('continue training...')
        return metrics


class SimpleDataframeDataset(Dataset):
    ### The __getitem__ method returns a dictionary with 'input' and 'label' keys
    def __init__(self, df, load_x_func):
        self.df = df
        self.load_x_func = load_x_func

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        load_x_func = self.load_x_func 
        x = load_x_func(self.df.iloc[idx])
        return {"input": x, "label": self.df.iloc[idx]['label']}

def mc_accuracy(logits, labels):
    preds = torch.argmax(logits, axis=1)
    accuracy = (preds == labels).float().mean()
    return {"accuracy": accuracy.item()}

class MultiClassClassifierTrainer:

    def __init__(self, model, train_df, trainer_config, load_x_func=None, batch_x_func=None,
                 exp_dir='./{self.__class__.__name__}-exp'):
        # model must have a forward method
        # model's output must be a tensor of shape (batch_size, num_classes)
        assert hasattr(model, 'forward'), "Model must have a forward method"

        self.model = model
        self.trainer_config = trainer_config
        self.load_x_func = load_x_func
        self.batch_x_func = batch_x_func
        self.exp_dir = exp_dir
        self.train_df = train_df
        if self.load_x_func is None:
            self.load_x_func = lambda x: torch.tensor(x['x']).float()

        self.training_metrics = {}


    def train(self):
        ts = time.time()
        
        train_df = self.train_df
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)

        # refine this later
        train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

        train_ds = SimpleDataframeDataset(train_df, self.load_x_func)
        test_ds = SimpleDataframeDataset(test_df, self.load_x_func)

        data_loader = DataLoader(train_ds, batch_size=100, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=100, shuffle=False)

        self.training_metrics['train_samples'] = len(train_df)
        self.training_metrics['test_samples'] = len(test_df)
        self.training_metrics['train_accuracy'] = []
        self.training_metrics['test_accuracy'] = []

        for epoch in range(10):
            def conquer_func_training(batch):
                x = batch['input']
                y = batch['label']
                o = self.model(x)
                loss = loss_fn(o, y)
                loss.backward()
                optimizer.step()
                accuracy = mc_accuracy(o, y)['accuracy']
                return {'num_samples': len(x), 'tp': accuracy*len(x)}
            
            train_accuracy = TorchBatchDivideAndConquer.divide_and_conquer(
                conquer_func=conquer_func_training,
                data_loader=data_loader,
                merger_func=lambda results: sum(r['tp'] for r in results) / sum(r['num_samples'] for r in results)
            )
            self.training_metrics['train_accuracy'].append(train_accuracy)

            # validation on test set

            def conquer_func_testing(batch):
                x = batch['input']
                y = batch['label']
                o = self.model(x)
                accuracy = mc_accuracy(o, y)['accuracy']
                return {'num_samples': len(x), 'tp': accuracy*len(x)}
            
            test_accuracy = TorchBatchDivideAndConquer.divide_and_conquer(
                conquer_func=conquer_func_testing,
                data_loader=test_loader,
                merger_func=lambda results: sum(r['tp'] for r in results) / sum(r['num_samples'] for r in results)
            )
            self.training_metrics['test_accuracy'].append(test_accuracy)

        self.training_metrics['train_time'] = time.time() - ts      
      

class MlpClassifier:

    def __init__(self, model_path: str):
        self.model_path = model_path

    def fit(self, train_data):

        pass

    def predict(self, test_data):
        pass


