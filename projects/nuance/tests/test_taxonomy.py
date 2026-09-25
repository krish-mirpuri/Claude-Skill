"""The label taxonomy and its two coarser views."""

from __future__ import annotations

import numpy as np
import pytest

from nuance.data.taxonomy import load_taxonomy


def test_every_fine_label_belongs_to_a_family(taxonomy):
    for name in taxonomy.fine:
        assert taxonomy.family_of(name, "ekman")
        assert taxonomy.family_of(name, "sentiment")


def test_neutral_survives_every_collapse(taxonomy):
    assert taxonomy.family_of("neutral") == "neutral"
    Y = np.zeros((1, len(taxonomy.fine)), dtype=np.int8)
    Y[0, taxonomy.index("neutral")] = 1
    for view in ("ekman", "sentiment"):
        out = taxonomy.collapse(Y, view)
        assert out[0, taxonomy.index("neutral", view)] == 1
        assert out.sum() == 1


def test_collapse_preserves_every_row(taxonomy):
    rng = np.random.default_rng(0)
    Y = (rng.random((200, len(taxonomy.fine))) < 0.1).astype(np.int8)
    Y[Y.sum(axis=1) == 0, 0] = 1
    for view in ("ekman", "sentiment"):
        assert (taxonomy.collapse(Y, view).sum(axis=1) > 0).all()


def test_collapse_merges_within_a_family(taxonomy):
    """Two labels from one family become one coarse label, not two."""
    Y = np.zeros((1, len(taxonomy.fine)), dtype=np.int8)
    Y[0, taxonomy.index("fear")] = 1
    Y[0, taxonomy.index("nervousness")] = 1
    assert taxonomy.collapse(Y, "ekman").sum() == 1


def test_collapse_keeps_distinct_families_apart(taxonomy):
    Y = np.zeros((1, len(taxonomy.fine)), dtype=np.int8)
    Y[0, taxonomy.index("joy")] = 1
    Y[0, taxonomy.index("anger")] = 1
    assert taxonomy.collapse(Y, "ekman").sum() == 2


def test_fine_view_is_the_identity(taxonomy):
    rng = np.random.default_rng(1)
    Y = (rng.random((50, len(taxonomy.fine))) < 0.2).astype(np.int8)
    np.testing.assert_array_equal(taxonomy.collapse(Y, "fine"), Y)


def test_projection_shapes(taxonomy):
    for view in ("fine", "ekman", "sentiment"):
        P = taxonomy.projection(view)
        assert P.shape == (len(taxonomy.fine), len(taxonomy.names(view)))
        assert (P.sum(axis=1) == 1).all(), "each fine label lands in exactly one group"


def test_unknown_view_is_rejected(taxonomy):
    with pytest.raises(ValueError):
        taxonomy.names("vibes")


def test_incomplete_mapping_is_rejected(raw_dir, tmp_path):
    """A taxonomy that forgets an emotion must fail loudly, not silently drop it."""
    import json
    import shutil

    shutil.copytree(raw_dir, tmp_path / "raw")
    mapping = json.loads((tmp_path / "raw" / "ekman_mapping.json").read_text())
    mapping["joy"] = [e for e in mapping["joy"] if e != "gratitude"]
    (tmp_path / "raw" / "ekman_mapping.json").write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="gratitude"):
        load_taxonomy(tmp_path / "raw")
