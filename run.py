"""Python script to find potential amplifier configurations."""

import collections
import itertools
from collections.abc import Iterable
from itertools import islice

import numpy as np

from amplifier import Amplifier


def sliding_window(iterable: Iterable, n: int) -> Iterable:
    "Collect data into overlapping fixed-length chunks or blocks."
    # sliding_window('ABCDEFG', 4) → ABCD BCDE CDEF DEFG
    it = iter(iterable)
    window = collections.deque(islice(it, n - 1), maxlen=n)
    for x in it:
        window.append(x)
        yield "".join(window)


def to_kilo(n: str) -> float:
    """Converts a string number into number in kilo (times 1000)."""
    return float(n) * 1000


STUDENT_ID = "20386501"
potential_resistors = itertools.chain(
    map(to_kilo, STUDENT_ID),
    map(to_kilo, sliding_window(STUDENT_ID, 2)),
    map(float, sliding_window(STUDENT_ID, 3)),
    map(float, sliding_window(STUDENT_ID, 4)),
    map(float, sliding_window(STUDENT_ID, 5)),
)
potential_resistors = np.unique(tuple(potential_resistors))
potential_resistors = np.fromiter(
    filter(lambda x: 100 <= x < 100e3, potential_resistors), dtype=float
)


if __name__ == "__main__":
    Q1 = (
        1e3,
        501,
        1e3,
        500,
    )
    Q2 = (
        6e3,
        8e3,
        2e3,
        3.8e3,
    )
    M3 = (1e3, 6.5e3, 80e3)
    amp = Amplifier(*Q1, *Q2, *M3)

    print(f"Q1: {amp.Q1_valid()}, V_C: {amp.V_C1}, V_B: {amp.V_B1}, V_E: {amp.V_E1}")
    print(f"Q2: {amp.Q2_valid()}, V_C: {amp.V_C2}, V_B: {amp.V_B2}, V_E: {amp.V_E2}")
    print(f"M3: {amp.M3_valid()}, V_D: {amp.V_D}, V_G: {amp.V_G}, V_S: {amp.V_S}")
    print(f"A_v: {amp.gain}")
