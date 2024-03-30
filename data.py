import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from typing import Dict
from transformers import BatchEncoding
from preprocessor import TextPreprocessor


class SatireDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, preprocessor: TextPreprocessor[BatchEncoding]):
        super().__init__()

        self.preprocessor: TextPreprocessor[BatchEncoding] = preprocessor
        self.dataset_processor, self.processed_text = self.preprocessor(dataset)

        self.classes = 'class' in self.dataset_processor.columns
        if self.classes:
            _, class_counts = np.unique(self.dataset_processor['class'], return_counts=True)
            class_counts = torch.from_numpy(class_counts)
            self.weights = 1 - class_counts / class_counts.sum()

    def __len__(self):
        return len(self.dataset_processor)

    def __getitem__(self, idx: int | slice) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask = (self.processed_text['input_ids'][idx], self.processed_text['attention_mask'][idx])

        if self.classes:
            return dict(
                label=torch.tensor(self.dataset_processor['class'].iloc[idx]),
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        else:
            return dict(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
