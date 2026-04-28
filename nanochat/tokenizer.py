"""
Tokenizer wrapper around the HuggingFace `tokenizers` library.
Loads the Llama 2 32k tokenizer from HuggingFace with custom special tokens.
"""

import os
from functools import lru_cache

from tokenizers import AddedToken
from tokenizers import Tokenizer as HFTokenizer

# HuggingFace model ID for the tokenizer (ungated mirror of Llama 2)
HF_TOKENIZER_ID = "NousResearch/Llama-2-7b-hf"

# Special tokens added on top of the Llama 2 base vocabulary.
# BOS uses Llama 2's native <s> (id=1) — no need for a custom <|bos|>.
# <|assistant_end|> is retained as a stop token recognised by the engine.
SPECIAL_TOKENS = [
    "<|assistant_end|>",
]


class Tokenizer:
    """Light wrapper around HuggingFace Tokenizer."""

    def __init__(self, tokenizer: HFTokenizer):
        self.tokenizer = tokenizer

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self) -> list[str]:
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        return [w.content for w in special_tokens_map.values()]

    def id_to_token(self, id: int) -> str:
        return self.tokenizer.id_to_token(id)

    def encode_special(self, text: str) -> int:
        token_id = self.tokenizer.token_to_id(text)
        assert token_id is not None, f"Special token {text!r} not found in tokenizer"
        return token_id

    def get_bos_token_id(self) -> int:
        # Llama 2's native BOS is <s> (id=1)
        return self.encode_special("<s>")

    def _encode_one(self, text: str, prepend: str | int | None = None, append: str | int | None = None) -> list[int]:
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode(self, text: str | list[str], prepend: str | int | None = None,
               append: str | int | None = None, num_threads: int = 8) -> list[int] | list[list[int]]:
        if isinstance(text, str):
            return self._encode_one(text, prepend=prepend, append=append)
        elif isinstance(text, list):
            # Use encode_batch for C++-level parallelism
            encodings = self.tokenizer.encode_batch(text, add_special_tokens=False)
            prepend_id = (prepend if isinstance(prepend, int) else self.encode_special(prepend)) if prepend is not None else None
            append_id = (append if isinstance(append, int) else self.encode_special(append)) if append is not None else None
            results = []
            for enc in encodings:
                ids = enc.ids
                if prepend_id is not None:
                    ids = [prepend_id] + ids
                if append_id is not None:
                    ids = ids + [append_id]
                results.append(ids)
            return results
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir: str) -> None:
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")


# -----------------------------------------------------------------------------
# nanochat-specific convenience functions


@lru_cache(maxsize=1)
def get_tokenizer() -> Tokenizer:
    tokenizer = HFTokenizer.from_pretrained(HF_TOKENIZER_ID)
    tokenizer.add_special_tokens([AddedToken(s, special=True) for s in SPECIAL_TOKENS])
    return Tokenizer(tokenizer)


