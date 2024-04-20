"""Amplifier classess and functions."""

# pylint: disable=invalid-name
from dataclasses import dataclass

import numpy as np

# Constants
V_T = 26e-3
# Circuit
V_CC = 15
R_S = 8e3
R_L = 300
# Transistor
BETA = 150
V_BE = 0.7
V_A = 65
# MOSFET
K_N = 5e-3
V_TN = -2
LAMBDA = 0.01


def par_res(*res: float) -> float:
    """Calculate the Resistance in Parallel."""
    return sum(map(lambda x: x**-1, res)) ** -1


@dataclass
class BJTAmplifier:
    """The BJT amplifier of the amplifier design."""

    R_1: float
    R_2: float
    R_C: float
    R_E: float

    @property
    def V_TH(self) -> float:
        """Thevanian Voltage of the BJT amplifier."""
        return V_CC * self.R_2 / (self.R_1 + self.R_2)

    @property
    def R_TH(self) -> float:
        """Thevanian Resistance of BJT amplifier."""
        return par_res(self.R_1, self.R_2)

    @property
    def I_B(self) -> float:
        """The base current of BJT amplifier."""
        return (self.V_TH - V_BE) / (self.R_TH + (1 + BETA) * self.R_E)

    @property
    def I_C(self) -> float:
        """The collector current of BJT amplifier."""
        return BETA * self.I_B

    @property
    def I_E(self) -> float:
        """The collector current of BJT amplifier."""
        return (1 + BETA) * self.I_B

    @property
    def V_B(self) -> float:
        """The base voltage of BJT amplifier."""
        return self.V_TH - self.I_B * self.R_TH

    @property
    def V_C(self) -> float:
        """The collector voltage of BJT amplifier."""
        return V_CC - self.I_C * self.R_C

    @property
    def V_E(self) -> float:
        """The collector voltage of the BJT amplifier."""
        return self.I_E * self.R_E

    @property
    def r_pi(self) -> float:
        """The r_pi of BJT amplifier."""
        return V_T / self.I_B

    @property
    def g_m(self) -> float:
        """The g_m of BJT amplifier."""
        return self.I_C / V_T

    @property
    def r_0(self) -> float:
        """The r_0 of BJT amplifier."""
        return V_A / self.I_C

    def is_valid(self):
        """Check if BJT amplifier is valid (forward active region) or not."""
        return self.V_E < self.V_B and self.V_B < self.V_C


@dataclass
class FETAmplifier:
    """The FET amplifier of the amplifier design."""

    R_1: float
    R_2: float
    R_SS: float

    @property
    def I_D(self) -> float:
        """The Drain current of M3."""
        a = K_N * self.R_SS**2
        b = -2 * K_N * self.R_SS * self.V_G + 2 * K_N * self.R_SS * V_TN - 1
        c = K_N * (V_TN**2 + self.V_G**2 - 2 * self.V_G * V_TN)

        # Applying Quadratic Equation
        I_D1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        I_D2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        # Checking if I_D is possible
        V_S1 = I_D1 * self.R_SS
        V_DS1 = self.V_D - V_S1
        V_GS1 = self.V_G - V_S1

        # Checking if V_DS1 > V_DS1(sat)
        # V_DS1(sat) = V_GS1 - V_TN
        if V_DS1 >= (V_GS1 - V_TN) and (V_GS1 > V_TN):
            return I_D1
        # Assuming the other one is the correct I_D if the first check failed
        return I_D2

    @property
    def V_G(self) -> float:
        """The Gate voltage of M3."""
        return V_CC * self.R_2 / (self.R_1 + self.R_2)

    @property
    def V_D(self) -> float:
        """The Drain voltage of M3."""
        return V_CC

    @property
    def V_S(self) -> float:
        """The Source voltage of M3."""
        return self.I_D * self.R_SS

    @property
    def g_m(self) -> float:
        """The g_m of M3."""
        return 2 * np.sqrt(K_N * self.I_D)

    @property
    def r_0(self) -> float:
        """The r_0 of M3."""
        return 1 / (LAMBDA * self.I_D)

    def is_valid(self):
        """Check if FET amplifier is valid (saturation region) or not."""
        V_GS = self.V_G - self.V_S
        V_DS = self.V_D - self.V_S
        return V_GS > V_TN and V_DS >= (V_GS - V_TN)


@dataclass
class Amplifier:
    """The amplifier design."""

    Q1: BJTAmplifier
    Q2: BJTAmplifier
    M3: FETAmplifier

    @property
    def gain(self) -> float:
        """The gain (A_v) of the amplifier."""
        n1 = self.Q1.g_m * self.Q2.g_m * self.M3.g_m
        n2 = par_res(R_L, self.M3.R_SS, self.M3.r_0)
        n3 = par_res(self.Q1.r_0, self.Q2.R_1, self.Q2.R_2, self.Q1.R_C, self.Q2.r_pi)
        n4 = par_res(self.Q2.r_0, self.M3.R_1, self.M3.R_2, self.Q2.R_C)
        n5 = par_res(self.Q1.R_1, self.Q1.R_2, self.Q1.r_pi)
        d1 = 1 + self.M3.g_m * par_res(R_L, self.M3.R_SS, self.M3.r_0)
        d2 = par_res(self.Q1.R_1, self.Q1.R_2, self.Q1.r_pi) + R_S
        return n1 * n2 * n3 * n4 * n5 / (d1 * d2)

    @property
    def R_in(self) -> float:
        """The input resistance (R_in) of the amplifier."""
        return par_res(self.Q1.R_1, self.Q1.R_2, self.Q1.r_pi)

    @property
    def R_out(self) -> float:
        """The output resistance (R_out) of the amplifier."""
        resistor = par_res(self.M3.r_0, self.M3.R_SS)
        return resistor / (1 + self.M3.g_m * resistor)

    def is_valid(self) -> bool:
        """Check if the amplifier is valid (all transitor working at the correct region)."""
        return self.Q1.is_valid() and self.Q2.is_valid() and self.M3.is_valid()
