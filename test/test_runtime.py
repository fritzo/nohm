import re

import pytest

from nohm.runtime import _gensym, parse, re_varname, readback, reduce, validate


def test_gensym():
    assert _gensym(0) == "a"
    assert _gensym(1) == "b"
    assert _gensym(25) == "z"
    assert _gensym(26) == "aa"
    assert _gensym(27) == "ab"


def normalize_text(text):
    """
    Normalizes text by removing comments, normalizing whitespace, and alpha
    converting.
    """
    text = re.sub("#.*[\n\r]", "", text)  # remove comments
    tokens = text.strip().split()
    rename = {}
    for i, token in enumerate(tokens):
        if re_varname.match(token):
            if token not in rename:
                rename[token] = _gensym(len(rename))
            tokens[i] = rename[token]
    return " ".join(tokens)


PARSE_EXAMPLES = [
    ("LAM a a", "LAM a a"),
    ("LAM a LAM b a", "LAM a LAM b a"),
    ("LAM a LAM b b", "LAM a LAM b b"),
    ("LAM a LAM a a", "LAM a LAM b b"),
    ("APP LAM a a LAM b b", "LAM a a"),
    (
        """
        LET one LAM f x APP f x
        LET two LAM f x APP f APP x x
        APP APP two two one
        """,
        "LAM a a",
    ),
    ("0", "LAM f LAM x x"),
    ("1", "LAM f LAM x APP f x"),
    ("2", "LAM f LAM x APP f APP f x"),
    ("3", "LAM f LAM x APP f APP f APP f x"),
]


@pytest.mark.parametrize("text,_", PARSE_EXAMPLES)
def test_parse_readback(text, _):
    expected = normalize_text(text)
    term = parse(text)
    validate(term)
    actual = readback(term)
    assert actual == expected


REDUCE_EXAMPLES = [
    ("LAM a a", "LAM a a"),
    ("APP LAM a a LAM b b", "LAM a a"),
    (
        """
        LET one LAM f x APP f x
        LET two LAM f x APP f APP x x
        APP APP two two one
        """,
        "LAM a a",
    ),
]


@pytest.mark.parametrize("text,redtext", REDUCE_EXAMPLES)
def test_reduce(text, redtext):
    term = parse(text)
    validate(term)
    port, term = reduce("out", term)
    validate(term)
    actual = readback(term)
    assert actual == redtext
