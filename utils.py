from typing import Tuple
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import pathlib as pb
import pandas as pd


def read_data(data_dir: pb.Path, train_filename: str = 'train.csv', test_filename: str = 'test.csv') -> (
        Tuple[pd.DataFrame, pd.DataFrame]):
    train_data_raw = pd.read_csv(data_dir / train_filename, sep=',', index_col=None, encoding='utf-8')
    test_data_raw = pd.read_csv(data_dir / test_filename, sep=',', index_col=None, encoding='utf-8')
    train_data_raw['class'] = train_data_raw['class'].map({True: 1, False: 0})

    train_data_raw['text'] = train_data_raw['title'].fillna('') + ' ' + train_data_raw['content'].fillna('')
    test_data_raw['text'] = test_data_raw['title'].fillna('') + ' ' + test_data_raw['content'].fillna('')

    train_data_raw = train_data_raw.drop(columns='id', errors='ignore')
    test_data_raw = test_data_raw.drop(columns='id', errors='ignore')

    return train_data_raw, test_data_raw


def compute_metrics(labels, preds):
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (tp / (tp + fn) + specificity) / 2
    return {
        'accuracy': accuracy_score(labels, preds),
        'balanced_accuracy': balanced_accuracy,
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds),
        'specificity': specificity
    }
