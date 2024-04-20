"""Amplifier classess and functions."""

# pylint: disable=invalid-name, too-many-instance-attributes, too-many-public-methods
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Constants
V_T = 26e-3
# Circuit
V_CC = 15
R_S = 8e3
R_L = 300
# Transistor
beta = 150
V_BE = 0.7
V_A = 65
# MOSFET
K_n = 5e-3
V_TN = -2
lambda_3 = 0.01


def par_res(*res: Tuple[float]) -> float:
    """Calculate the Resistance in Parallel."""
    return sum(map(lambda x: x**-1, res)) ** -1 # type: ignore


@dataclass
class BJTAmplifier:
    """The BJT amplifier of the Amplifier Design."""
    R_1: float
    R_2: float
    R_C: float
    R_E: float

    @property
    def V_TH(self) -> float:
        """Thevanian Voltage of the BJT transistor."""
        return V_CC * self.R_2 / (self.R_1 + self.R_2)

    @property
    def R_TH(self) -> float:
        """Thevanian Resistance of Q1."""
        return par_res(self.R_1, self.R_2)

    @property
    def I_B(self) -> float:
        """The base current of Q1."""
        return (self.V_TH - V_BE) / (self.R_TH + (1 + beta) * self.R_E)

    @property
    def I_C(self) -> float:
        """The collector current of Q1."""
        return beta * self.I_B

    @property
    def I_E(self) -> float:
        """The collector current of Q1."""
        return (1 + beta) * self.I_B

    @property
    def V_B(self) -> float:
        """The base voltage of Q1."""
        return self.V_TH - self.I_B * self.R_TH

    @property
    def V_C(self) -> float:
        """The collector voltage of Q1."""
        return V_CC - self.I_C * self.R_C

    @property
    def V_E(self) -> float:
        """The collector voltage of Q1."""
        return self.I_E * self.R_E



@dataclass
class Amplifier:
    """The amplifier design."""

    # Q1 Parameters
    R_11: float
    R_21: float
    R_C1: float
    R_E1: float

    # Q2 Parameters
    R_12: float
    R_22: float
    R_C2: float
    R_E2: float

    # M3 Parameters
    R_13: float
    R_23: float
    R_SS: float

    @property
    def V_TH1(self) -> float:
        """Thevanian Voltage of Q1."""
        return V_CC * self.R_21 / (self.R_11 + self.R_21)

    @property
    def R_TH1(self) -> float:
        """Thevanian Resistance of Q1."""
        return par_res(self.R_11, self.R_21)

    @property
    def I_B1(self) -> float:
        """The base current of Q1."""
        return (self.V_TH1 - V_BE) / (self.R_TH1 + (1 + beta) * self.R_E1)

    @property
    def I_C1(self) -> float:
        """The collector current of Q1."""
        return beta * self.I_B1

    @property
    def I_E1(self) -> float:
        """The collector current of Q1."""
        return (1 + beta) * self.I_B1

    @property
    def V_B1(self) -> float:
        """The base voltage of Q1."""
        return self.V_TH1 - self.I_B1 * self.R_TH1

    @property
    def V_C1(self) -> float:
        """The collector voltage of Q1."""
        return V_CC - self.I_C1 * self.R_C1

    @property
    def V_E1(self) -> float:
        """The collector voltage of Q1."""
        return self.I_E1 * self.R_E1

    @property
    def V_TH2(self) -> float:
        """Thevanian Voltage of Q2."""
        return V_CC * self.R_22 / (self.R_12 + self.R_22)

    @property
    def R_TH2(self) -> float:
        """Thevanian Resistance of Q2."""
        return par_res(self.R_12, self.R_22)

    @property
    def I_B2(self) -> float:
        """The base current of Q2."""
        return (self.V_TH2 - V_BE) / (self.R_TH2 + (1 + beta) * self.R_E2)

    @property
    def I_C2(self) -> float:
        """The collector current of Q2."""
        return beta * self.I_B2

    @property
    def I_E2(self) -> float:
        """The collector current of Q2."""
        return (1 + beta) * self.I_B2

    @property
    def V_B2(self) -> float:
        """The base voltage of Q2."""
        return self.V_TH2 - self.I_B2 * self.R_TH2

    @property
    def V_C2(self) -> float:
        """The collector voltage of Q2."""
        return V_CC - self.I_C2 * self.R_C2

    @property
    def V_E2(self) -> float:
        """The collector voltage of Q2."""
        return self.I_E2 * self.R_E2

    @property
    def I_D(self) -> float:
        """The Drain current of M3."""
        a = K_n * self.R_SS**2
        b = -2 * K_n * self.R_SS * self.V_G + 2 * K_n * self.R_SS * V_TN - 1 
        c = K_n * (V_TN**2 + self.V_G**2 - 2 * self.V_G * V_TN)

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
        return V_CC * self.R_23 / (self.R_13 + self.R_23)

    @property
    def V_D(self) -> float:
        """The Drain voltage of M3."""
        return V_CC

    @property
    def V_S(self) -> float:
        """The Source voltage of M3."""
        return self.I_D * self.R_SS

    @property
    def r_pi1(self) -> float:
        """The r_pi of Q1."""
        return V_T / self.I_B1

    @property
    def g_m1(self) -> float:
        """The g_m of Q1."""
        return self.I_C1 / V_T

    @property
    def r_01(self) -> float:
        """The r_0 of Q1."""
        return V_A / self.I_C1

    @property
    def r_pi2(self) -> float:
        """The r_pi of Q2."""
        return V_T / self.I_B2

    @property
    def g_m2(self) -> float:
        """The g_m of Q2."""
        return self.I_C2 / V_T

    @property
    def r_02(self) -> float:
        """The r_0 of Q2."""
        return V_A / self.I_C2

    @property
    def g_m3(self) -> float:
        """The g_m of M3."""
        return 2 * np.sqrt(K_n * self.I_D)

    @property
    def r_03(self) -> float:
        """The r_0 of M3."""
        return 1 / (lambda_3 * self.I_D)

    @property
    def gain(self) -> float:
        """The gain (A_v) of the amplifier."""
        n1 = self.g_m1 * self.g_m2 * self.g_m3
        n2 = par_res(R_L, self.R_SS, self.r_03)
        n3 = par_res(self.r_01, self.R_12, self.R_22, self.R_C1, self.r_pi2)
        n4 = par_res(self.r_02, self.R_13, self.R_23, self.R_C2)
        n5 = par_res(self.R_11, self.R_21, self.r_pi1)
        d1 = 1 + self.g_m3 * par_res(R_L, self.R_SS, self.r_03)
        d2 = par_res(self.R_11, self.R_21, self.r_pi1) + R_S
        return n1 * n2 * n3 * n4 * n5 / (d1 * d2)

    def Q1_valid(self) -> bool:
        """Check if Q1 is valid (forward active region) or not."""
        return self.V_E1 < self.V_B1 and self.V_B1 < self.V_C1

    def Q2_valid(self) -> bool:
        """Check if Q2 is valid (forward active region) or not."""
        return self.V_E2 < self.V_B2 and self.V_B2 < self.V_C2

    def M3_valid(self) -> bool:
        """Check if M3 is valid (saturation region) or not."""
        V_GS = self.V_G - self.V_S
        V_DS = self.V_D - self.V_S
        return V_GS > V_TN and V_DS >= (V_GS - V_TN)

    def is_valid(self) -> bool:
        """Check if the amplifier is valid (all transitor working at the correct region)."""
        return self.Q1_valid() and self.Q2_valid() and self.M3_valid()
