from typing import List

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

__all__ = ["fit_tokenizer"]


def fit_tokenizer(
    corpus: List[str],
    vocab_size: int = 30522,
    batch_size: int = 1000,
) -> PreTrainedTokenizerFast:
    # Creating Byte-Pair Encoding tokenizer
    raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    raw_tokenizer.normalizer = normalizers.Sequence([])
    raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    def batch_gen():
        """
        A generator function for iterating over a dataset in chunks.
        """
        for i in range(0, len(corpus), batch_size):
            yield corpus[i : i + batch_size]

    raw_tokenizer.train_from_iterator(batch_gen(), trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
