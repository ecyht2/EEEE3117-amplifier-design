"""Python script to find potential amplifier configurations."""

import collections
import itertools
from collections.abc import Iterable
from itertools import islice

import numpy as np

from amplifier import Amplifier, BJTAmplifier, FETAmplifier


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
    Q1 = BJTAmplifier(
        1e3,
        501,
        1e3,
        500,
    )
    Q2 = BJTAmplifier(
        6e3,
        8e3,
        2e3,
        3.8e3,
    )
    M3 = FETAmplifier(1e3, 6.5e3, 80e3)
    amp = Amplifier(Q1, Q2, M3)

    print("Transistor Current")
    print(f"Q1: I_B: {Q1.I_B:.3g}, I_C: {Q1.I_C:.3g}, I_E: {Q1.I_E:.3g}")
    print(f"Q2: I_B: {Q2.I_B:.3g}, I_C: {Q2.I_C:.3g}, I_E: {Q2.I_E:.3g}")
    print(f"M3: I_D: {M3.I_D:.3g}")

    print("Transistor Validity")
    print(f"Q1: {Q1.is_valid()}, V_C: {Q1.V_C}, V_B: {Q1.V_B}, V_E: {Q1.V_E}")
    print(f"Q2: {Q2.is_valid()}, V_C: {Q2.V_C}, V_B: {Q2.V_B}, V_E: {Q2.V_E}")
    print(f"M3: {M3.is_valid()}, V_D: {M3.V_D}, V_G: {M3.V_G}, V_S: {M3.V_S}")

    print("AC Values")
    print(f"Q1: g_m: {Q1.g_m:.3f}, r_0: {Q1.r_0:.3f}, r_pi: {Q1.r_pi:.3f}")
    print(f"Q2: g_m: {Q2.g_m:.3f}, r_0: {Q2.r_0:.3f}, r_pi: {Q2.r_pi:.3f}")
    print(f"M3: g_m: {M3.g_m:.3f}, r_0: {M3.r_0:.3f}")

    print("AC Analysis")
    print(f"A_v: {amp.gain}")
    print(f"R_in: {amp.R_in}")
    print(f"R_out: {amp.R_out}")
