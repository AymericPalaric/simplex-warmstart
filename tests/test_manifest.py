import pytest

from simplex_warmstart.manifest import next_batch_entry, summarise

BATCHES = [{"id": 0, "n_studies": 120, "seed": 0, "families": ["esters"]}]


def test_new_batch_gets_the_next_id():
    entry = next_batch_entry(BATCHES, 40, ["esters"])
    assert entry["id"] == 1


def test_seed_is_derived_and_distinct():
    first = next_batch_entry(BATCHES, 40, ["esters"])
    second = next_batch_entry([*BATCHES, first], 40, ["esters"])
    assert first["seed"] != second["seed"]


def test_unknown_family_is_refused():
    with pytest.raises(ValueError, match="Unknown"):
        next_batch_entry(BATCHES, 40, ["kryptonite"])


def test_empty_batch_is_refused():
    with pytest.raises(ValueError):
        next_batch_entry(BATCHES, 0, ["esters"])


def test_summary_counts_studies_across_batches():
    entry = next_batch_entry(BATCHES, 40, ["silicones"])
    stats = summarise([*BATCHES, entry])
    assert stats == {"n_batches": 2, "n_studies": 160, "families": ["esters", "silicones"]}
