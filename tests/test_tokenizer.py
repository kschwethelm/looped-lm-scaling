"""Tests for the Llama 2 32k tokenizer wrapper."""

import pytest
from nanochat.tokenizer import SPECIAL_TOKENS, Tokenizer, get_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer()


def test_vocab_size_and_bos(tokenizer):
    assert tokenizer.get_vocab_size() == 32001  # 32000 base + 1 special (<|assistant_end|>)
    assert tokenizer.get_bos_token_id() == 1  # Llama 2's native <s>


def test_special_tokens(tokenizer):
    ids = [tokenizer.encode_special(t) for t in SPECIAL_TOKENS]
    assert ids == list(range(32000, 32001))


def test_roundtrip(tokenizer):
    text = "Hello! I'm testing 123 + Unicode: 你好 🌍\ndef foo():\n    pass"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_batch_matches_single(tokenizer):
    texts = ["hello world", "foo bar", "12345"]
    assert tokenizer.encode(texts) == [tokenizer.encode(t) for t in texts]


def test_prepend_append(tokenizer):
    ids = tokenizer.encode("hi", prepend="<s>", append="<|assistant_end|>")
    assert ids[0] == 1
    assert ids[-1] == tokenizer.encode_special("<|assistant_end|>")
