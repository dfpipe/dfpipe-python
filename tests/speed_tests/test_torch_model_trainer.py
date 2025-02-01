
import pytest
import pandas as pd
from pathlib import Path
from dfpipe.torch.trainer import TorchModelTrainer
from dfpipe.torch.data import DFDataset
import torch
import numpy as np

def test_torch_model_trainer():
    # this test case is to test the torch model trainer with a task of classifying a point in a 10-dimensional space
    # generate a dataframe with x, 10 features and y
    d = 10
    x = torch.randn(1000, d)
    y = (x[:, 0] * x[:, 1] > 0).long().tolist()
    df = pd.DataFrame({'x': x.tolist(), 'y': y})

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = torch.nn.Sequential(
        torch.nn.Linear(d, d),
        torch.nn.ReLU(),
        torch.nn.Linear(d, 2),
    ).to(device)

    def forward_function(x):
        return model(x)

    def epoch_end_callback(metrics):
        print(metrics)

    model_parameters = model.parameters()

    ds = DFDataset.xy_dataset(df)

    trainer = TorchModelTrainer()
    metrics =  trainer.fit(model_parameters, forward_function, ds, epoch_num=10, optimizer_name='adamw', device=device, loss_name='cross_entropy', epoch_end_callback=epoch_end_callback)
    epoch = metrics['epoch'][-1]
    loss = np.mean(metrics['loss'][epoch])
    batch_accuracy = np.mean(metrics['batch_accuracy'][epoch])
    assert loss <= 1
    assert epoch == 9
    assert batch_accuracy > 0.5

    
def test_torch_model_trainer_speed(benchmark):
    result = benchmark(test_torch_model_trainer)
    assert result is None

