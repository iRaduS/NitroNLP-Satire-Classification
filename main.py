import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random

from torch import optim
from pathlib import Path

from torch.utils.data import DataLoader
from utils import read_data
from preprocessor import BERTPreprocessor, RoBERTPreprocessor, MT5Preprocessor
from data import SatireDataset
from models import RoBERTFlatClassModel, MT5FlatClassModel, BERTFlatClassModel, Args, Output
from train import train, evaluate

train_data_frame, test_data_frame \
    = read_data(Path('data'), 'train.csv', 'test.csv')

DEVICE: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
SEED = 61

bert_preprocessor = BERTPreprocessor()
train_dataset = SatireDataset(train_data_frame, bert_preprocessor)
random.seed(SEED)
np.random.seed(SEED)
np.random.RandomState(SEED)
torch.manual_seed(SEED)

model_setup = [
    {
        'name': 'bert',
        'model_factory': lambda: BERTFlatClassModel().to(DEVICE),
        'optim_factory': lambda params: optim.AdamW(params, lr=Args().learning_rate, weight_decay=Args().weight_decay),
        'loss_fn': nn.CrossEntropyLoss(weight=train_dataset.weights.to(DEVICE)),
        'train_dataset': SatireDataset(train_data_frame, bert_preprocessor),
        'test_dataset': SatireDataset(test_data_frame, bert_preprocessor),
    },
    # {
    #     'name': 'mt5',
    #     'model_factory': lambda: MT5FlatClassModel().to(DEVICE),
    #     'optim_factory': lambda params: optim.AdamW(params, lr=Args().learning_rate, weight_decay=Args().weight_decay),
    #     'loss_fn': nn.CrossEntropyLoss(weight=train_dataset.weights),
    #     'train_dataset': SatireDataset(train_data_frame, MT5Preprocessor()),
    #     'test_dataset': SatireDataset(test_data_frame, MT5Preprocessor()),
    # },
    # {
    #     'name': 'RoBERT',
    #     'model_factory': lambda: RoBERTFlatClassModel().to(DEVICE),
    #     'optim_factory': lambda params: optim.AdamW(params, lr=Args().learning_rate, weight_decay=Args().weight_decay),
    #     'loss_fn': nn.BCELoss(weight=train_dataset.weights),
    #     'train_dataset': SatireDataset(train_data_frame, RoBERTPreprocessor()),
    #     'test_dataset': SatireDataset(test_data_frame, RoBERTPreprocessor()),
    # }
]

current_model_idx = 0
args = Args()
model = model_setup[current_model_idx]['model_factory']()
train_dataset = model_setup[current_model_idx]['train_dataset']
test_dataset = model_setup[current_model_idx]['test_dataset']
optimizer = model_setup[current_model_idx]['optim_factory'](model.parameters())
loss_fn = model_setup[current_model_idx]['loss_fn']

# Create the dataloaders
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)

# --- Training ---
train_output: Output = train(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_loader=train_loader,
    args=args,
    device=DEVICE,
)
print('Last Epoch - Loss: {}, Accuracy: {}'.format(train_output.loss_mean, train_output.accy_mean))

test_loader: DataLoader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
test_output: Output = evaluate(
    model=model,
    loss_fn=loss_fn,
    data_loader=test_loader,
    with_labels=False,
    device=DEVICE
)

output: pd.DataFrame = pd.DataFrame({'class': pd.Series(data=test_output.predictions_as_indx)})
output = output.reset_index()
output = output.rename(columns={'index': 'id'})
output.to_csv('submission.csv', index=False)