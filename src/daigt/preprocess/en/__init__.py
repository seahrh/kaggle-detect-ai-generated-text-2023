from typing import AnyStr, Set

import numpy as np
import scml
import scml.nlp as snlp
import spacy

from daigt.preprocess import Preprocessor

__all__ = [
    "BasicPreprocessor",
    "BowPreprocessor",
    "digit_frac",
    "letter_frac",
    "space_frac",
    "punc_frac",
    "upper_frac",
    "repeat_char_frac",
    "repeat_substring_frac",
    "sentence_length_std",
    "sentence_length_mean",
    "sentence_length_delta_std",
    "sentence_length_delta_mean",
]

log = scml.get_logger(__name__)
_r_char = snlp.RepeatingCharacter(max_times=3, letters=True, punctuation=True)
_r_substring = snlp.RepeatingSubstring(
    min_length=3, max_times=1, letters=True, punctuation=True
)


def digit_frac(s: str) -> float:
    return snlp.count_digit(s) / len(s)  # type: ignore


def letter_frac(s: str) -> float:
    return snlp.count_alpha(s) / len(s)  # type: ignore


def space_frac(s: str) -> float:
    return snlp.count_space(s) / len(s)  # type: ignore


def punc_frac(s: str) -> float:
    return snlp.count_punctuation(s) / len(s)  # type: ignore


def upper_frac(s: str) -> float:
    return snlp.count_upper(s) / len(s)  # type: ignore


def repeat_char_frac(s: str) -> float:
    return _r_char.count(s) / len(s)  # type: ignore


def repeat_substring_frac(s: str) -> float:
    return _r_substring.count_char(s) / len(s)  # type: ignore


def sentence_length_mean(s: str) -> float:
    sents = snlp.sentences(s)
    sents = [len(s.split()) for s in sents]
    return float(np.mean(sents))


def sentence_length_std(s: str) -> float:
    sents = snlp.sentences(s)
    if len(sents) < 2:
        return 0
    sents = [len(s.split()) for s in sents]
    return float(np.std(sents))


def sentence_length_delta_mean(s: str) -> float:
    sents = snlp.sentences(s)
    if len(sents) < 2:
        return 0
    sents = [len(s.split()) for s in sents]
    deltas = []
    for i in range(1, len(sents)):
        deltas.append(abs(sents[i] - sents[i - 1]))
    return float(np.mean(deltas))


def sentence_length_delta_std(s: str) -> float:
    sents = snlp.sentences(s)
    if len(sents) < 3:
        return 0
    sents = [len(s.split()) for s in sents]
    deltas = []
    for i in range(1, len(sents)):
        deltas.append(abs(sents[i] - sents[i - 1]))
    return float(np.std(deltas))


class BasicPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()

    def __call__(self, s: AnyStr, **kwargs) -> str:
        res: str = snlp.to_ascii(s)
        res = snlp.collapse_whitespace(res)
        return res


class BowPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm", exclude=["textcat", "ner", "tok2vec"])
        log.debug(self.nlp.pipe_names)
        self.stops: Set[str] = self.nlp.Defaults.stop_words
        log.debug(f"{len(self.stops)} stopwords\n{sorted(list(self.stops))}")
        self.contraction = snlp.ContractionExpansion()

    def __call__(self, s: AnyStr, drop_stopword: bool = False, **kwargs) -> str:
        res: str = snlp.to_ascii(s)
        res = self.contraction.apply(res)
        # Expand contractions before removing punctuation
        res = snlp.strip_punctuation(res, replacement=" ")
        doc = self.nlp(res)
        tokens = []
        for token in doc:
            # some lemma has uppercase char
            t = token.lemma_.lower()
            if drop_stopword and t in self.stops:
                continue
            tokens.append(t)
        res = " ".join(tokens)
        res = snlp.collapse_whitespace(res)
        return res
