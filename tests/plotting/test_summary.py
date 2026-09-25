"""Tests for the text and summary helpers the plots share."""

import pytest

from mmmjax.plotting._summary import _shorten, _wrap


def test_wrap_breaks_long_labels_after_separators_within_the_width():
    label = "social_media_meta_dynamic_brand_world_cup"

    result = _wrap(label, 18)

    assert result == "social_media_meta_\ndynamic_brand_\nworld_cup"
    assert result.replace("\n", "") == label
    assert max(len(line) for line in result.split("\n")) <= 18


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("TV", "TV"),
        ("programmatic", "programmatic"),
        ("unbreakablechannelnamewithoutseparators", "unbreakablechannelnamewithoutseparators"),
        ("Linear TV prime time", "Linear TV\nprime time"),
    ],
)
def test_wrap_leaves_short_or_unbreakable_labels_whole(label, expected):
    result = _wrap(label, 12)

    assert result == expected


@pytest.mark.parametrize(
    ("label", "limit", "expected"),
    [
        ("social_media_meta_dynamic_brand_world_cup", 27, "social_media_\u2026and_world_cup"),
        ("social_media_meta_dynamic_brand_world_cup", 12, "socia\u2026ld_cup"),
        ("search", 27, "search"),
    ],
)
def test_shorten_keeps_both_ends_of_a_long_label(label, limit, expected):
    result = _shorten(label, limit)

    assert result == expected
    assert len(result) == min(len(label), limit)
