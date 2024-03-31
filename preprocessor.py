# import spacy
from pandas import DataFrame
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic, Tuple
from transformers import AutoTokenizer, BatchEncoding
from tqdm import tqdm
# from spacy.lang.ro import Romanian


PT = TypeVar('PT')
URL_REGEX = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
NUMBER_REGEX = r"\d+"
SPACE_REGEX = r"\s+"
SPECIAL_CHARS_REGEX = r"[&#.,?!\"\'‘’“”]|(\([^)]*\))"
HTML_REGEX = r'<[^<]+?>'


class TextPreprocessor(ABC, Generic[PT]):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, dataset: DataFrame) -> PT:
        raise NotImplemented()


class AutoPreprocessor(TextPreprocessor[BatchEncoding]):
    def __init__(self, repo: str, cased: bool = True, max_length: int = 64) -> None:
        super().__init__()

        self.repo: str = repo
        self.cased = cased

        self.tokenizer = AutoTokenizer.from_pretrained(self.repo)
        self.max_length: int = max_length
    #     self.nlp = spacy.load('ro_core_news_lg')
    #
    # def lemmatize(self, text: str) -> str:
    #     doc = self.nlp(text)
    #     return " ".join([token.lemma_.strip() for token in doc])

    def __call__(self, dataset: DataFrame) -> Tuple[DataFrame, BatchEncoding]:
        dataset = dataset.drop(dataset[dataset['text'].str.contains('\t')].index)
        dataset = dataset.reset_index()

        dataset['text'] = dataset['text'].str.replace(
            URL_REGEX, '', regex=True)
        dataset['text'] = dataset['text'].str.replace(
            NUMBER_REGEX, '', regex=True)
        dataset['text'] = dataset['text'].str.replace(
            SPECIAL_CHARS_REGEX, '', regex=True)
        dataset['text'] = dataset['text'].str.replace(
            HTML_REGEX, '', regex=True)
        dataset['text'] = dataset['text'].str.replace(
            SPACE_REGEX, ' ', regex=True)

        if not self.cased:
            dataset['text'] = dataset['text'].str.lower()

        # Manual preprocessing
        sentences: List[str] = []
        for text in tqdm(dataset['text'].tolist()):
            text: str = text \
                .replace("ţ", "ț") \
                .replace("ş", "ș") \
                .replace("Ţ", "Ț") \
                .replace("Ş", "Ș") \
                .replace('”', '"') \
                .replace('„', '"') \
                .replace('“', '"') \
                .replace('­', '') \
                .replace('–', '') \
                .replace('—', '')
            # lemmatized_text = self.lemmatize(text)
            sentences.append(lemmatized_text)

        return dataset, self.tokenizer.__call__(
            sentences,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )


class BERTPreprocessor(AutoPreprocessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(repo='dumitrescustefan/bert-base-romanian-cased-v1',
                         *args, **kwargs)