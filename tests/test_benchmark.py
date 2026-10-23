"""Tests: python -m pytest -q"""
from pathlib import Path

import pytest

from netvisbench.common import load_items
from netvisbench.generate import build
from netvisbench.oracle import solve
from netvisbench.scoring import extract_answer, is_correct


@pytest.fixture(scope="module")
def items(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    build(out, seed=7, per_difficulty=6)
    return out, load_items(out)


def test_counts(items):
    _, its = items
    assert len(its) == 6 * 4 * 6  # families x levels x per_difficulty
    assert len({i.id for i in its}) == len(its)


def test_images_exist(items):
    out, its = items
    for it in its:
        assert (Path(out) / it.image).stat().st_size > 1000


def test_oracle_agrees_with_ground_truth(items):
    _, its = items
    wrong = [(i.id, solve(i), i.answer) for i in its if solve(i) != i.answer]
    assert not wrong, wrong


def test_generation_is_deterministic(tmp_path):
    a = build(tmp_path / "a", seed=11, per_difficulty=2)
    b = build(tmp_path / "b", seed=11, per_difficulty=2)
    assert [x.answer for x in a] == [y.answer for y in b]
    assert [x.text_rendering for x in a] == [y.text_rendering for y in b]


@pytest.mark.parametrize("raw,atype,expected", [
    ("The cost is 12.\nANSWER: 12", "int", "12"),
    ("ANSWER: **B**", "choice", "B"),
    ("Answer: h3", "label", "H3"),
    ("so the broadcast is 10.0.0.255\nANSWER: 10.0.0.255.", "label", "10.0.0.255"),
    ("I think it's 7 hops... final answer 9", "int", "9"),
])
def test_extract(raw, atype, expected):
    assert extract_answer(raw, atype) == expected


def test_is_correct():
    assert is_correct("A", "A", "choice")
    assert is_correct("0012", "12", "int")
    assert not is_correct(None, "12", "int")
