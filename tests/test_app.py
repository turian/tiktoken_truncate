#!/usr/bin/env python
"""Tests for `tiktoken_truncate` package."""
# pylint: disable=redefined-outer-name

import random

import pytest
import tiktoken

from tiktoken_truncate.globals import model_max_tokens
from tiktoken_truncate.random_string import random_string
from tiktoken_truncate.slow import (
    truncate_document_to_max_tokens as truncate_document_to_max_tokens_slow,
)
from tiktoken_truncate.tiktoken_truncate import (
    get_avg_tokens_per_char,
    truncate_document_to_max_tokens,
)

# Fixed seed for deterministic tests
FIXED_SEED = 0

N_TESTS = 30


def generate_test_data(seed, ntests=N_TESTS // 2):
    rng = random.Random(seed)
    models = list(model_max_tokens.keys())
    test_data = []
    for _ in range(ntests):
        model = rng.choice(models)
        max_tokens = model_max_tokens[model]
        encoding = tiktoken.encoding_for_model(model)
        estimated_characters = max_tokens / get_avg_tokens_per_char(encoding=encoding)
        k = int(estimated_characters * rng.uniform(0.5, 2.0))
        text = random_string(k=k, seed=rng.randint(0, 2**32))
        test_data.append((model, text))
    return test_data


# Parametrize with deterministic test data
deterministic_test_data = generate_test_data(FIXED_SEED)


@pytest.mark.parametrize("model,text", deterministic_test_data)
def test_slow_vs_fast_deterministic(model, text):
    text_slow = truncate_document_to_max_tokens_slow(text=text, model=model)
    text_fast = truncate_document_to_max_tokens(text=text, model=model)
    assert text_slow == text_fast


# Parametrize with random test data for broader coverage
random_test_data = generate_test_data(random.randint(0, 2**32))


@pytest.mark.parametrize("model,text", random_test_data)
def test_slow_vs_fast_random(model, text):
    text_slow = truncate_document_to_max_tokens_slow(text=text, model=model)
    text_fast = truncate_document_to_max_tokens(text=text, model=model)
    assert text_slow == text_fast


if __name__ == "__main__":
    pytest.main()
