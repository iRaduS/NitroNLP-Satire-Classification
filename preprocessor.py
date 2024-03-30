from pandas import DataFrame
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic, Tuple
from transformers import AutoTokenizer, BatchEncoding
from tqdm import tqdm

BAIT_WORDS = [
    "ALERT(Ă|A)", "alerta", "Alertă",
    "ANALIZA", "analiza", "Analiza",
    "Ancheta", "ANCHETA", "ancheta",
    "BREAKING NEWS", "Breaking News", "breaking news",
    "CAPTURA", "captura", "Captura",
    "CLIP", "clip", "Clip",
    "CONFLICT", "conflict", "Conflict",
    "CONFRUNTARE", "confruntare", "Confruntare",
    "CONTROVERSAT", "controversat", "Controversat",
    "CRIZ(Ă|A)", "criză", "Criză",
    "DECLARAȚIE", "declarație", "Declarație",
    "DEZASTU", "dezastru", "Dezastru",
    "DEZVALUIRI", "dezvaluiri", "Dezvaluiri",
    "DRAM(Ă|A)", "dram(ă|a)", "Dramă",
    "EXCLUSIV", "exclusiv", "Exclusiv",
    "EXCLUSIVITATE", "exclusivitate", "Exclusivitate",
    "EXPLOZIV", "exploziv", "Exploziv",
    "EXTRAORDINAR", "extraordinar", "Extraordinar",
    "FOTO", "foto", "Foto",
    "GALERIE", "galerie", "Galerie",
    "IMAGINI", "imagini", "Imagini",
    "INCENDIAR", "incendiar", "Incendiar",
    "INREGISTRARE", "inregistrare", "Inregistrare",
    "INTERVIU", "interviu", "Interviu",
    "LIVE", "live", "Live",
    "MISTER", "mister", "Mister",
    "PROVOCARE", "provocare", "Provocare",
    "REPLAY", "replay", "Replay",
    "REPORTAJ", "reportaj", "Reportaj",
    "REVELAȚIE", "revelație", "Revelație",
    "SCANDAL", "scandal", "Scandal",
    "SENZAȚIONAL", "senzațional", "Senzațional",
    "ȘOC", "șoc", "ŞOC", "Soc", "soc", "SOC",
    "TRAGEDIE", "tragedie", "Tragedie",
    "UIMITOR", "uimitor", "Uimitor",
    "ULTIM(Ă|A) OR(Ă|A)", "Ultim(ă|a) or(ă|a)", "ultima ora",
    "URGENT", "urgent", "Urgent",
    "VIDEO", "video", "Video"
]

PT = TypeVar('PT')
URL_REGEX = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
NUMBER_REGEX = r"\d+"
SPACE_REGEX = r"\s+"
SPECIAL_CHARS_REGEX = r"[&#.,?!\"\'‘’“”]|(\([^)]*\))"
BAIT_WORDS_REGEX = r'\b(' + '|'.join(BAIT_WORDS) + r')\b'
EMAIL_REGEX = r'\S*@\S*\s?'
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

    def __call__(self, dataset: DataFrame) -> Tuple[DataFrame, BatchEncoding]:
        dataset = dataset.drop(dataset[dataset['text'].str.contains('\t')].index)
        dataset = dataset.reset_index()

        dataset['text'] = dataset['text'].str.replace(
            URL_REGEX, '', regex=True)
        dataset['text'] = dataset['text'].str.replace(
            NUMBER_REGEX, '', regex=True)
        # dataset['text'] = dataset['text'].str.replace(
        #     SPECIAL_CHARS_REGEX, '', regex=True)
        # dataset['text'] = dataset['text'].str.replace(
        #     BAIT_WORDS_REGEX, '', regex=True)
        # dataset['text'] = dataset['text'].str.replace(
        #     EMAIL_REGEX, '', regex=True)
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
            sentences.append(text)

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


class MT5Preprocessor(AutoPreprocessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(repo='dumitrescustefan/mt5-base-romanian',
                         *args, **kwargs)


class RoBERTPreprocessor(AutoPreprocessor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(repo='readerbench/RoBERT-base',
                         cased=False,
                         *args, **kwargs)
