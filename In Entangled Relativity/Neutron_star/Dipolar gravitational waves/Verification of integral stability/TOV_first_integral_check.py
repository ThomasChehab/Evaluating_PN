#!/usr/bin/env python
"""TOV integration (Entangled Relativity) with a first-integral consistency check.

This file is based on the provided `TOV.py` and adds a diagnostic based on the
first integral that follows from Eq. (20) of PhysRevD.103.024034 together with
the polytropic EoS Eq. (30):

    P = K rho^gamma,   with gamma = 5/3.

In the code, `rho` is the *mass* density (SI), so the energy density is e = rho*c^2.
With rho = (P/k)^(3/5), Eq. (20) implies

    d ln a + 2 dP/(P + e) - (Lm - P)/(P + e) d ln Phi = 0,

where a = g_tt, Phi = φ, and Lm is the chosen on-shell matter Lagrangian.

For gamma=5/3 with the above EoS, the enthalpy integral is analytic:

    2 \int dP/(P+e) = 5 ln( A + P^{2/5} ) + const,

with A = c^2 / k^{3/5}.

Therefore, defining

    C(r) = ln a(r) + 5 ln(A + P(r)^{2/5}) - \int^r (Lm-P)/(P+e) d ln Phi,

one should have C(r) ≈ constant along the numerical integration.

Special cases:
  - Lm = P    -> last integral vanishes.
  - Lm = -e   -> last integral = -\int d ln Phi, so C = ln a + 5 ln(...) + ln Phi.

Running this file will execute a small demo and print the max drift of C(r).
"""

import scipy.constants as cst
import numpy as np
from scipy.integrate import solve_ivp

# SciPy renamed cumtrapz -> cumulative_trapezoid in recent versions
try:
    from scipy.integrate import cumtrapz as integcum  # type: ignore
except ImportError:  # pragma: no cover
    from scipy.integrate import cumulative_trapezoid as integcum  # type: ignore


c2 = cst.c**2
kappa = 8 * np.pi * cst.G / c2**2

# Polytropic constant used in the original file (Eq. 30, with unit conversions)
k = 1.475e-3 * (cst.fermi**3 / (cst.eV * 1e6)) ** (2 / 3) * c2 ** (5 / 3)

massSun = 1.989e30


# Equation of state: P = k rho^(5/3)
def PEQS(rho):
    return k * rho ** (5 / 3)


def RhoEQS(P):
    return (P / k) ** (3 / 5)


def Lagrangian(P, option):
    """On-shell matter Lagrangian choice.

    option == 0: Lm = T = -rho*c^2 + 3P
    option == 1: Lm = -rho*c^2
    option == 2: Lm = P
    """
    rho = RhoEQS(P)
    if option == 0:
        return -c2 * rho + 3 * P
    elif option == 1:
        return -c2 * rho
    elif option == 2:
        return P
    raise ValueError("option must be 0 (T), 1 (-rho), or 2 (P)")


def b(r, m):
    return (1 - (c2 * m * kappa / (4 * np.pi * r))) ** (-1)


def adota(r, P, m, Psi, Phi):
    A = (b(r, m) / r)
    B = (
        1
        - (1 / b(r, m))
        + P * kappa * r**2 * Phi ** (-1 / 2)
        - 2 * r * Psi / (b(r, m) * Phi)
    )
    C = (1 + r * Psi / (2 * Phi)) ** (-1)
    return A * B * C


def D00(r, P, m, Psi, Phi, option):
    ADOTA = adota(r, P, m, Psi, Phi)
    rho = RhoEQS(P)
    Lm = Lagrangian(P, option)
    T = -c2 * rho + 3 * P
    A = Psi * ADOTA / (2 * Phi * b(r, m))
    B = kappa * (Lm - T) / (3 * Phi ** (1 / 2))
    return A + B


def bdotb(r, P, m, Psi, Phi, option):
    rho = RhoEQS(P)
    A = -b(r, m) / r
    B = 1 / r
    C = b(r, m) * r * (
        -D00(r, P, m, Psi, Phi, option) + kappa * c2 * rho * Phi ** (-1 / 2)
    )
    return A + B + C


def f1(r, P, m, Psi, Phi, option):
    ADOTA = adota(r, P, m, Psi, Phi)
    Lm = Lagrangian(P, option)
    rho = RhoEQS(P)
    return -(ADOTA / 2) * (P + rho * c2) + (Psi / (2 * Phi)) * (Lm - P)


def f2(r, P, m, Psi, Phi, option):
    rho = RhoEQS(P)
    A = 4 * np.pi * rho * (Phi ** (-1 / 2)) * r**2
    B = 4 * np.pi * (-D00(r, P, m, Psi, Phi, option) / (kappa * c2)) * r**2
    return A + B


def f4(r, P, m, Psi, Phi, option, dilaton_active):
    ADOTA = adota(r, P, m, Psi, Phi)
    BDOTB = bdotb(r, P, m, Psi, Phi, option)
    rho = RhoEQS(P)
    Lm = Lagrangian(P, option)
    T = -c2 * rho + 3 * P
    A = (-Psi / 2) * (ADOTA - BDOTB + 4 / r)
    B = b(r, m) * kappa * Phi ** (1 / 2) * (T - Lm) / 3
    if dilaton_active:
        return A + B
    return 0.0


def f3(r, P, m, Psi, Phi, option, dilaton_active):
    if dilaton_active:
        return Psi
    return 0.0


def dy_dr(r, y, option, dilaton_active):
    P, M, Phi, Psi = y
    return [
        f1(r, P, M, Psi, Phi, option),
        f2(r, P, M, Psi, Phi, option),
        f3(r, P, M, Psi, Phi, option, dilaton_active),
        f4(r, P, M, Psi, Phi, option, dilaton_active),
    ]


def first_integral_profile(r, P, M, Phi, Psi, option):
    """Compute the first-integral constant profile C(r).

    Returns:
      C(r): array
      drift: (C - C[0])
    """
    r = np.asarray(r)
    P = np.asarray(P)
    Phi = np.asarray(Phi)
    Psi = np.asarray(Psi)

    # Build ln a(r) from a'/a = adota, with ln a(r_min)=0.
    F1 = adota(r, P, M, Psi, Phi)
    ln_a = np.concatenate([[0.0], integcum(F1, r)])

    # Analytic enthalpy piece for gamma=5/3 with energy density e = c2*(P/k)^(3/5)
    A = c2 / (k ** (3 / 5))
    # guard against tiny/negative P due to numerical overshoot near the surface
    P_clip = np.clip(P, 1e-60, None)
    enthalpy_piece = 5.0 * np.log(A + P_clip ** (2 / 5))

    # Phi-dependent integral term
    rho = RhoEQS(P_clip)
    e = rho * c2
    denom = P_clip + e
    Lm = np.array([Lagrangian(p, option) for p in P_clip])
    integrand = (Lm - P_clip) / denom * (Psi / Phi)  # = (Lm-P)/(P+e) * d ln Phi / dr
    Iphi = np.concatenate([[0.0], integcum(integrand, r)])

    C = ln_a + enthalpy_piece - Iphi
    drift = C - C[0]
    return C, drift


class TOV:
    def __init__(
        self,
        initDensity,
        initPsi=0.0,
        initPhi=1.0,
        radiusMax_in=3.0e4,
        Npoint=4000,
        option=2,
        dilaton_active=True,
    ):
        self.initDensity = initDensity
        self.initPressure = PEQS(initDensity)
        self.initPsi = initPsi
        self.initPhi = initPhi
        self.initMass = 0.0
        self.option = option
        self.dilaton_active = dilaton_active

        self.radiusMax_in = radiusMax_in
        self.Npoint = Npoint

        self.sol = None
        self.first_integral = None
        self.first_integral_drift = None

    def compute(self):
        r_min = 1e-9
        r_eval = np.linspace(r_min, self.radiusMax_in, self.Npoint)
        y0 = [self.initPressure, self.initMass, self.initPhi, self.initPsi]

        sol = solve_ivp(
            dy_dr,
            [r_min, self.radiusMax_in],
            y0,
            method="RK45",
            t_eval=r_eval,
            rtol=1e-8,
            atol=1e-10,
            args=(self.option, self.dilaton_active),
        )
        self.sol = sol

        P, M, Phi, Psi = sol.y
        C, drift = first_integral_profile(sol.t, P, M, Phi, Psi, self.option)
        self.first_integral = C
        self.first_integral_drift = drift
        return sol


def _demo_run():
    # A simple run intended as a sanity check of the first integral.
    # (The original code prints initDensity in MeV/fm^3; here we just reuse the same input convention.)
    initDensity = 100.0

    for option, name in [(2, "Lm=P"), (1, "Lm=-rho c^2"), (0, "Lm=T")]:
        tov = TOV(
            initDensity=initDensity,
            initPsi=0.0,
            initPhi=1.0,
            radiusMax_in=3.0e4,
            Npoint=6000,
            option=option,
            dilaton_active=True,
        )
        sol = tov.compute()
        drift = tov.first_integral_drift
        max_abs = float(np.nanmax(np.abs(drift)))
        # normalize by typical scale of C to get a relative measure
        C0 = float(tov.first_integral[0])
        rel = max_abs / max(1.0, abs(C0))
        print(f"[{name}] status={sol.status}, steps={sol.t.size}, max|ΔC|={max_abs:.3e}, rel≈{rel:.3e}")


if __name__ == "__main__":
    _demo_run()
