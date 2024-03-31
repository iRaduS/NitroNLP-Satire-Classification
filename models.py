import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, BertModel
from typing import Dict, List
from abc import ABC, abstractmethod
import numpy as np


class Args(object):
    def __init__(self) -> None:
        self.num_epochs = 3
        self.batch_size = 64
        self.weight_decay = 5e-6
        self.learning_rate = 2e-5


class Output(object):
    def __init__(self, predictions: List[int], loss_seq: List[float], accy_seq: List[float],
                 with_labels: bool) -> None:
        self.predictions_as_indx: np.ndarray = np.array(predictions)
        self.with_labels: bool = with_labels
        if self.with_labels:
            self.loss_seq: np.ndarray = np.array(loss_seq)
            self.accy_seq: np.ndarray = np.array(accy_seq)
            self.loss_mean: float = self.loss_seq.mean()
            self.accy_mean: float = self.accy_seq.mean()


class PretrainedFlatClassModel(ABC, nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def create_layers(self) -> nn.Sequential:
        raise NotImplementedError()

    def unfreeze(self, layers: str | List[str] | List[nn.Module], unfreeze: bool = True) -> None:
        if isinstance(layers, list) and isinstance(layers[0], nn.Module):
            for module in layers:
                for param in module.parameters():
                    param.requires_grad = unfreeze
        elif layers == 'all':
            for param in self.parameters():
                param.requires_grad = unfreeze
        elif layers == 'none':
            self.requires_grad_(False)
            return
        elif isinstance(layers[0], str):
            for (param_name, param) in self.named_parameters():
                if param_name in layers:
                    param.requires_grad = unfreeze
        else:
            raise ValueError('invalid layers param - bad unfreeze')


class BERTFlatClassModel(PretrainedFlatClassModel):
    def __init__(self,
                 dropout: float = 0.1,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = None
        self.dropout: float = dropout
        self.n_classes = 2
        self.repo = 'dumitrescustefan/bert-base-romanian-cased-v1'
        self.bert_model: BertModel = AutoModel.from_pretrained(self.repo)
        self.unfreeze([
            self.bert_model.pooler.dense,
            self.bert_model.encoder.layer[-1:]
        ])
        self.create_layers()

    def create_layers(self) -> None:
        self.layers = nn.Sequential(
            nn.Linear(in_features=768, out_features=512),
            nn.GELU(),
            nn.Linear(in_features=512, out_features=self.n_classes)
        )

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        input_ids: Tensor = x['input_ids']
        attention_mask: Tensor = x['attention_mask']
        _, output = self.bert_model(
            input_ids, attention_mask, return_dict=False)
        output = self.layers(output)

        return output


class BERTLSTMClassModel(PretrainedFlatClassModel):
    def __init__(self,
                 dropout: float = 0.1,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lstm = None
        self.classifier = None
        self.dropout: float = dropout
        self.n_classes = 2
        self.num_layers = 2
        self.repo = 'dumitrescustefan/bert-base-romanian-cased-v1'
        self.bert_model: BertModel = AutoModel.from_pretrained(self.repo)
        self.unfreeze([
            self.bert_model.pooler.dense,
            self.bert_model.encoder.layer[-1:]
        ])
        self.create_layers()

    def create_layers(self) -> None:
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=256,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=self.dropout if self.num_layers > 1 else 0)

        self.classifier = nn.Linear(
            in_features=256 * 2,
            out_features=self.n_classes
        )

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        input_ids: Tensor = x['input_ids']
        attention_mask: Tensor = x['attention_mask']
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        bert_output = outputs.last_hidden_state

        lstm_output, (hidden_state, cell_state) = self.lstm(bert_output)
        lstm_output = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)

        output = self.classifier(lstm_output)
        return output


class Ensemble(nn.Module):
    def __init__(self, models: List[Dict[str, str | nn.Module]], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Init models
        self.models = models

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        y_hats = []

        for model in self.models:
            if model['name'] == 'bert':
                y_hat = model['model'](**{k: v for k, v in x.items() if k != 'label'}).logits.detach().cpu()
            else:
                y_hat = model['model'].forward(x).detach().cpu()
            y_hats.append(y_hat)

        stacked = torch.stack(y_hats, dim=1)
        predictions = torch.sum(stacked, dim=1)
        return predictions
