# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 9: ``generator.allocate_mix`` (SPEC.md section 11).

``allocate_mix(count)`` splits ``count`` generated scenarios 30/25/25/20 into
``(happy, edge, adversarial, hallucination)`` with largest-remainder rounding. It is deterministic,
and ties are broken in that order (happy, edge, adversarial, hallucination). ``allocate_mix(11)``
is ``(3, 3, 3, 2)``.
"""

import pytest

WEIGHTS = (30, 25, 25, 20)  # percent: happy, edge, adversarial, hallucination


def _allocate_mix(count):
    from impulse_evals.generator import allocate_mix

    return allocate_mix(count)


def reference_mix(count):
    """Largest-remainder rounding in integer arithmetic (no float error); ties in list order."""
    floors = [count * weight // 100 for weight in WEIGHTS]
    remainders = [count * weight % 100 for weight in WEIGHTS]
    left = count - sum(floors)
    for index in sorted(range(len(WEIGHTS)), key=lambda i: (-remainders[i], i))[:left]:
        floors[index] += 1
    return tuple(floors)


def test_allocate_mix_eleven_is_3_3_3_2():
    assert _allocate_mix(11) == (3, 3, 3, 2)


def test_allocate_mix_returns_a_tuple_of_four_ints():
    mix = _allocate_mix(11)
    assert isinstance(mix, tuple)
    assert len(mix) == 4
    assert all(type(part) is int for part in mix)


@pytest.mark.parametrize(
    "count, expected",
    [
        (1, (1, 0, 0, 0)),
        (2, (1, 1, 0, 0)),
        (3, (1, 1, 1, 0)),
        (4, (1, 1, 1, 1)),
        (5, (2, 1, 1, 1)),
        (7, (2, 2, 2, 1)),
        (10, (3, 3, 2, 2)),  # edge and adversarial tie on .5: edge first
        (11, (3, 3, 3, 2)),
        (20, (6, 5, 5, 4)),
        (30, (9, 8, 7, 6)),
        (40, (12, 10, 10, 8)),
    ],
)
def test_allocate_mix_hand_computed_values(count, expected):
    assert _allocate_mix(count) == expected


@pytest.mark.parametrize("count", range(1, 41))
def test_allocate_mix_sums_to_count_and_matches_largest_remainder(count):
    mix = _allocate_mix(count)
    assert sum(mix) == count
    assert all(part >= 0 for part in mix)
    assert mix == reference_mix(count)


def test_allocate_mix_is_deterministic():
    first = [_allocate_mix(count) for count in range(1, 41)]
    second = [_allocate_mix(count) for count in range(1, 41)]
    assert first == second


def test_reference_mix_self_check():
    """The reference used above agrees with the SPEC example."""
    assert reference_mix(11) == (3, 3, 3, 2)
