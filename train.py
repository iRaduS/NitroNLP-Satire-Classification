import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from models import Args, Output
from typing import Dict, List
from sklearn.metrics import balanced_accuracy_score


def train(model: nn.Module, optimizer: Optimizer, loss_fn: nn.Module, train_loader: DataLoader, args: Args, device: torch.device, use_pretrained = False):
    for epoch in range(args.num_epochs):
        model.train()

        epoch_loss: List[float] = []
        epoch_accuracy: List[float] = []
        predictions: List[int] = []
        for batch in tqdm(train_loader):
            batch: Dict[str, torch.Tensor] = {k: v.to(device) for k, v in batch.items()}

            y_true: torch.Tensor | np.ndarray = batch['label']
            if use_pretrained:
                y_pred: torch.Tensor | np.ndarray = model(**{k: v for k, v in batch.items() if k != 'label'}).logits
            else:
                y_pred: torch.Tensor | np.ndarray = model.forward(batch)

            optimizer.zero_grad()
            loss: torch.Tensor = loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()

            y_true = y_true.detach().cpu().numpy()
            y_pred = y_pred.detach().argmax(dim=1).cpu().numpy()

            predictions.extend(y_pred.tolist())
            epoch_accuracy.append(balanced_accuracy_score(y_true, y_pred))
            epoch_loss.append(loss.detach().cpu().numpy())

            output = Output(predictions, epoch_loss, epoch_accuracy, with_labels=True)
            print('Train Epoch {} - Loss: {}, Balanced accuracy: {}'.format(epoch, output.loss_mean, output.accy_mean))
    return output


def evaluate(model: nn.Module, loss_fn: nn.Module, data_loader: DataLoader,
             with_labels: bool = True, device: torch.device = torch.device('cpu')) -> Output:
    model.eval()
    with torch.no_grad():
        epoch_loss: List[float] = []
        epoch_accy: List[float] = []
        predictions: List[int] = []

        for batch in tqdm(data_loader):
            batch: Dict[str, torch.Tensor] = {k: v.to(device) for k, v in batch.items()}

            y_pred: torch.Tensor | np.ndarray = model.forward(batch)
            if with_labels:
                y_true: torch.Tensor | np.ndarray = batch['label']
                loss: torch.Tensor = loss_fn(y_pred, y_true)

            y_pred = y_pred.detach().argmax(dim=1).cpu().numpy()
            predictions.extend(y_pred.tolist())

            if with_labels:
                y_true = y_true.detach().cpu().numpy()
                epoch_accy.append(balanced_accuracy_score(y_true, y_pred))
                epoch_loss.append(loss.detach().cpu().numpy())

    output = Output(predictions, epoch_loss, epoch_accy, with_labels=with_labels)
    if with_labels:
        print('Validation Epoch - Loss: {}, Balanced accuracy: {}'.format(output.loss_mean, output.accy_mean))
    return output