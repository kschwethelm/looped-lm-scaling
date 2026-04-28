import random

from dev.eval_bundles.owned.prepare_owned_bundle import generate_induction_head_example


def test_induction_head_example_structure():
    rng = random.Random(0)
    ex = generate_induction_head_example(rng, n_pairs=6)
    assert "context" in ex and "continuation" in ex
    assert ex["continuation"].strip() in ex["context"]
    # n_pairs demo lines + 1 query line, all containing "->"
    lines = [l for l in ex["context"].split("\n") if "->" in l]
    assert len(lines) == 7
    # Query line is last and ends with " ->"
    assert ex["context"].rstrip().endswith("->")
