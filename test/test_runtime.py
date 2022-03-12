import gc
from collections import Counter

import pytest

from nohm.runtime import Term, collect, gensym, parse, readback, reduce, validate


def get_term_stats():
    return Counter(type(o) for o in gc.get_objects() if isinstance(o, Term))


def test_gensym():
    assert gensym(0) == "a"
    assert gensym(1) == "b"
    assert gensym(2) == "c"
    assert gensym(25) == "z"
    assert gensym(26) == "aa"
    assert gensym(27) == "ab"
    assert gensym(28) == "ac"
    assert gensym(26 + 26**2) == "aaa"


PARSE_EXAMPLES = [
    ("BOT", "BOT"),
    ("TOP", "TOP"),
    ("APP BOT TOP", "APP BOT TOP"),
    ("LAM a BOT", "LAM a BOT"),
    ("LAM a TOP", "LAM a TOP"),
    ("LAM a a", "LAM a a"),
    ("LAM a APP a a", "LAM a APP a a"),
    ("LAM a APP a APP a a", "LAM a APP a APP a a"),
    ("LAM a LAM b a", "LAM a LAM b a"),
    ("LAM a LAM b b", "LAM a LAM b b"),
    ("LAM a LAM a a", "LAM a LAM b b"),
    ("LAM a LAM b APP a b", "LAM a LAM b APP a b"),
    ("LAM a LAM b JOIN a b", "LAM a LAM b JOIN a b"),
    ("APP LAM a a LAM a a", "APP LAM a a LAM b b"),
    ("LET bot BOT LAM x x", "LAM a a"),
    ("LET bot BOT bot", "BOT"),
    ("LET bot BOT APP bot bot", "APP BOT BOT"),
    ("LET one LAM x x one", "LAM a a"),
    ("LET one LAM x x LET zero LAM f LAM x x one", "LAM a a"),
    ("LET zero LAM f LAM x x LET one LAM x x one", "LAM a a"),
    ("0", "LAM a LAM b b"),
    ("1", "LAM a LAM b APP a b"),
    ("2", "LAM a LAM b APP a APP a b"),
    ("3", "LAM a LAM b APP a APP a APP a b"),
    ("IMPORT lib ok", "LAM a a"),
    ("IMPORT lib true", "LAM a LAM b a"),
    ("IMPORT lib false", "LAM a LAM b b"),
]


@pytest.mark.parametrize("text,expected", PARSE_EXAMPLES)
def test_parse_readback(text, expected):
    main = parse(text)
    validate(main)
    actual = readback(main)
    assert actual == expected

    # Check for memory leaks.
    collect(main)
    del main
    counts = get_term_stats()
    assert not counts, counts


REDUCE_EXAMPLES = [
    ("BOT", "BOT"),
    ("TOP", "TOP"),
    ("APP BOT BOT", "BOT"),
    ("APP BOT TOP", "BOT"),
    ("APP TOP BOT", "TOP"),
    ("APP TOP TOP", "TOP"),
    ("LAM a a", "LAM a a"),
    ("LAM x BOT", "LAM a BOT"),
    ("LAM x TOP", "LAM a TOP"),
    ("LAM x LAM y APP x y", "LAM a LAM b APP a b"),
    ("LAM x LAM y JOIN x y", "LAM a LAM b JOIN a b"),
    ("APP LAM a a LAM b b", "LAM a a"),
    pytest.param(
        """
        LET one LAM f LAM x APP f x
        LET two LAM f LAM x APP f APP f x
        APP APP two two one
        """,
        "LAM a a",
        marks=[pytest.mark.xfail(reason="TODO")],
    ),
]


@pytest.mark.parametrize("text,expected", REDUCE_EXAMPLES)
def test_reduce(text, expected):
    main = parse(text)
    validate(main)
    reduce(main)
    validate(main)
    actual = readback(main)
    assert actual == expected

    # Check for memory leaks.
    collect(main)
    del main
    counts = get_term_stats()
    assert not counts, counts
