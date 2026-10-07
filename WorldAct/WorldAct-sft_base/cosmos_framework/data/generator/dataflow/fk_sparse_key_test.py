"""The sparse ``fk`` key must keep one slot per sample through the collate path.

Only some episodes carry an FK annotation, so a packed group can mix samples
that have ``sample["fk"]`` with samples that do not.  The accumulator builds its
per-sample list by first-appearance, so without a None backfill the list comes
out shorter than the group -- silently misaligned against ``sequence_plan``
rather than raising.

Drives the real ``VFMListCollator.collate`` entry, so the collate-stage
normalization and the ``_accumulate`` backfill are both under test.  The
final check re-runs the pre-fix accumulation loop and asserts it still
misaligns, which is what gives the assertions above their teeth.

Run:  PYTHONPATH=. <venv>/bin/python cosmos_framework/data/generator/dataflow/fk_sparse_key_test.py
"""

from types import SimpleNamespace

import torch

from cosmos_framework.data.generator.dataflow.collators import (
    VFMListCollator,
    _split_one,
    _vfm_inner_collate,
)

KEYPOINTS = 21


def make_sample(with_fk: bool) -> dict:
    """A minimal stand-in for one dataset sample.

    ``video`` is what the backfill measures the group length with, so it has to
    be present on every sample and reach ``output_batch`` on the first one.
    """
    sample = {
        "video": torch.randn(2, 8, 8, 3),
        "text_token_ids": torch.tensor([1, 2, 3]),
        "sequence_plan": [SimpleNamespace(has_fk=with_fk)],
        "domain_id": [0],
    }
    if with_fk:
        sample["fk"] = {
            "inputs": {"point_ids": list(range(KEYPOINTS))},
            "targets": {},
            "metadata": {},
        }
    return sample


def run_case(flags) -> dict:
    return VFMListCollator().collate([make_sample(f) for f in flags])


def main():
    # Annotated first, then two unannotated: a naive append stops at length 1.
    batch = run_case([True, False, False])
    fk = batch["fk"]
    assert len(fk) == 3, f"expected 3 slots, got {len(fk)}"
    assert fk[0] is not None and fk[1] is None and fk[2] is None, fk
    assert len(batch["sequence_plan"]) == 3
    print(f"✅ 带 FK 在前、不带在后 -> len(fk) = {len(fk)}  [有, None, None]")

    # Unannotated first: the naive first-appearance starts the list a sample late.
    batch = run_case([False, True, True])
    fk = batch["fk"]
    assert len(fk) == 3, f"expected 3 slots, got {len(fk)}"
    assert fk[0] is None and fk[1] is not None and fk[2] is not None, fk
    assert len(batch["sequence_plan"]) == 3
    print(f"✅ 不带在前、带 FK 在后 -> len(fk) = {len(fk)}  [None, 有, 有]")

    # A group with no FK at all must not grow an ``fk`` key out of nothing.
    batch = run_case([False, False])
    assert "fk" not in batch, "an FK-free group must not produce an 'fk' key"
    print("✅ 整组无 FK -> 不产生 'fk' 键")

    # Teeth: the pre-fix loop, run over the real splits, still misaligns.
    broken = _accumulate_without_guard([True, False, False])
    assert len(broken) == 1, f"guard-free path should misalign to 1, got {len(broken)}"
    print(f"🔬 去掉回填（对照）-> len(fk) = {len(broken)}  ← 正是被修掉的错位")
    print("\nPASS")


def _accumulate_without_guard(flags) -> list:
    """Verbatim pre-fix ``_accumulate`` over the real ``_split_one`` output."""
    output_batch: dict = {}
    for flag in flags:
        split = _split_one(_vfm_inner_collate([make_sample(flag)]))
        for key, value in split.items():
            if key not in output_batch:
                output_batch[key] = [value]
            else:
                output_batch[key].append(value)
    return output_batch["fk"]


if __name__ == "__main__":
    main()
