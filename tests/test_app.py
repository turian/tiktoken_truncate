#!/usr/bin/env python
"""Tests for `tiktoken_truncate` package."""
# pylint: disable=redefined-outer-name

import random

import pytest
import tiktoken
from tqdm import tqdm

from tiktoken_truncate.globals import model_max_tokens
from tiktoken_truncate.random_string import random_string
from tiktoken_truncate.slow import (
    truncate_document_to_max_tokens as truncate_document_to_max_tokens_slow,
)
from tiktoken_truncate.tiktoken_truncate import (
    get_avg_tokens_per_char,
    truncate_document_to_max_tokens,
)


def test_slow_vs_fast():
    NTESTS = 100
    rng = random.Random()
    rng.seed(0)
    models = list(model_max_tokens.keys())
    for i in tqdm(list(range(NTESTS))):
        model = rng.choice(models)
        max_tokens = model_max_tokens[model]
        encoding = tiktoken.encoding_for_model(model)
        estimated_characters = max_tokens / get_avg_tokens_per_char(encoding=encoding)
        k = int(estimated_characters * rng.uniform(0.5, 2.0))
        # k = int(estimated_characters * rng.uniform(0.5, 0.75))
        text = random_string(k=k, seed=rng.randint(0, 2**32))
        print(len(text))
        text_slow = truncate_document_to_max_tokens_slow(text=text, model=model)
        print(len(text))
        text_fast = truncate_document_to_max_tokens(text=text, model=model)
        assert text_slow == text_fast


if __name__ == "__main__":
    pytest.main()
