from typing import AnyStr, Set

import scml
import scml.nlp as snlp
import spacy

from daigt.preprocess import Preprocessor

__all__ = ["BasicPreprocessor", "BowPreprocessor"]

log = scml.get_logger(__name__)


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
