import os.path

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random

from torch import optim
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from utils import read_data
from tqdm import tqdm
from preprocessor import BERTPreprocessor
from data import SatireDataset
from models import BERTFlatClassModel, Args, Output, BERTLSTMClassModel, Ensemble
from train import train, evaluate

train_data_frame, test_data_frame \
    = read_data(Path('data'), 'train.csv', 'test.csv')

HAS_K_FOLD_ACTIVE: bool = False
DEVICE: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
SEED = 37
random.seed(SEED)
np.random.seed(SEED)
np.random.RandomState(SEED)
torch.manual_seed(SEED)

bert_preprocessor = BERTPreprocessor()
train_dataset = SatireDataset(train_data_frame, bert_preprocessor)
test_dataset = SatireDataset(test_data_frame, bert_preprocessor)

model_setup = [
    {
        'name': 'bert-mlp',
        'model_factory': lambda: BERTFlatClassModel().to(DEVICE),
        'optim_factory': lambda params: optim.AdamW(params, lr=Args().learning_rate, weight_decay=Args().weight_decay),
        'loss_fn': nn.CrossEntropyLoss(),
    },
    {
        'name': 'bert-lstm',
        'model_factory': lambda: BERTLSTMClassModel().to(DEVICE),
        'optim_factory': lambda params: optim.AdamW(params, lr=Args().learning_rate, weight_decay=Args().weight_decay),
        'loss_fn': nn.CrossEntropyLoss(weight=train_dataset.weights.to(DEVICE)),
    },
    {
        'name': 'bert',
        'model_factory': lambda: AutoModelForSequenceClassification.from_pretrained(
            'dumitrescustefan/bert-base-romanian-cased-v1', num_labels=2).to(DEVICE),
        'optim_factory': lambda params: optim.AdamW(params, lr=Args().learning_rate, weight_decay=Args().weight_decay),
        'loss_fn': nn.CrossEntropyLoss(weight=train_dataset.weights.to(DEVICE)),
    },
]

pretrained_models = []
args = Args()
for value in tqdm(model_setup):
    model = value['model_factory']()
    if os.path.exists(f'{value["name"]}_weights.pth'):
        model.load_state_dict(torch.load(f'{value["name"]}_weights.pth'))
        pretrained_models.append({'name': value["name"], 'model': model})
        continue

    optimizer = value['optim_factory'](model.parameters())
    loss_fn = value['loss_fn']

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    train_output: Output = train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        args=args,
        device=DEVICE,
        use_pretrained=(True if value["name"] == "bert" else False)
    )
    print('Last Epoch - Loss: {}, Accuracy: {}'.format(train_output.loss_mean, train_output.accy_mean))
    torch.save(model.state_dict(), f'{value["name"]}_weights.pth')
    pretrained_models.append({'name': value["name"], 'model': model})

model = Ensemble(pretrained_models)
test_loader: DataLoader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
test_output: Output = evaluate(
    model=model,
    loss_fn=nn.CrossEntropyLoss(),
    data_loader=test_loader,
    with_labels=False,
    device=DEVICE
)

output: pd.DataFrame = pd.DataFrame({'class': pd.Series(data=test_output.predictions_as_indx)})
output = output.reset_index()
output = output.rename(columns={'index': 'id'})
output.to_csv('submission.csv', index=False)
