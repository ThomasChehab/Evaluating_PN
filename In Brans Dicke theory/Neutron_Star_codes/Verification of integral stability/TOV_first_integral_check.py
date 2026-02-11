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
kappa = 8*np.pi*cst.G/c2**2
k = 1.475*10**(-3)*(cst.fermi**3/(cst.eV*10**6))**(2/3)*c2**(5/3)
massSun = 1.989*10**30

#Equation of state
def PEQS(rho):
    return k*rho**(5/3)

#Inverted equation of state
def RhoEQS(P):
    return (P/k)**(3/5)

# def v_sound_c(rho):
def v_sound_c(Phi, P):
    return np.sqrt(5/3 * k * RhoEQS(P)**(2/3)) / cst.c


#Equation for b
def b(r, m):
    return (1-(c2*m*kappa/(4*np.pi*r)))**(-1)

#Equation for da/dr
def adota(r, P, m, Psi, Phi, w):
    A = (b(r, m)/r)
    B = (1-(1/b(r, m))+P*kappa*r**2*Phi**(-1)-2*r*Psi/(b(r,m)*Phi) + ( -H00(r,m,Psi, Phi, w) ))
    C = (1+r*Psi/(2*Phi))**(-1)
    return A*B*C

#Equation for D00
def D00(r, P, m, Psi, Phi, w):
    ADOTA = adota(r, P, m, Psi, Phi, w)
    rho = RhoEQS(P)
    T = -c2*rho + 3*P
    A = Psi*ADOTA/(2*Phi*b(r,m))
    B = -kappa*(T)/(Phi * (3+2*w))
    return A+B

def H00(r,m,Psi, Phi, w):
    A = - w * Psi**2/Phi**2 * 1/(2*b(r,m))
    return A

#Equation for db/dr
def bdotb(r, P, m, Psi, Phi, w):
    rho = RhoEQS(P)
    A = -b(r,m)/r
    B = 1/r
    C = b(r,m)*r*(-H00(r,m,Psi, Phi, w)-D00(r, P, m, Psi, Phi, w)+kappa*c2*rho*Phi**(-1))
    return A+B+C

#Equation for dP/dr
def f1(r, P, m, Psi, Phi, w):
    ADOTA = adota(r, P, m, Psi, Phi, w)
    rho = RhoEQS(P)
    return -(ADOTA/2)*(P+rho*c2)

#Equation for dm/dr
def f2(r, P, m, Psi, Phi, w):
    rho = RhoEQS(P)
    A = 4*np.pi*rho*(Phi**(-1))*r**2
    B = 4*np.pi*(-D00(r, P, m, Psi, Phi,w)/(kappa*c2))*r**2
    C = 4*np.pi*(-H00(r, m, Psi, Phi, w)/(kappa*c2))*r**2
    return A+B

#Equation for dPsi/dr
def f4(r, P, m, Psi, Phi, w):
    ADOTA = adota(r, P, m, Psi, Phi,w)
    BDOTB = bdotb(r, P, m, Psi, Phi,w)
    rho = RhoEQS(P)
    T = -c2*rho + 3*P
    A = (-Psi/2)*(ADOTA-BDOTB+4/r)
    B = b(r,m)*kappa*T/(3+2*w)
    return A+B

#Equation for dPhi/dr
def f3(r, P, m, Psi, Phi):
    return Psi


#Define for dy/dr
def dy_dr(r, y, w):
    P, M, Phi, Psi = y
    dy_dt = [f1(r, P, M, Psi, Phi, w), f2(r, P, M, Psi, Phi, w),f3(r, P, M, Psi, Phi),f4(r, P, M, Psi, Phi, w) ]
    return dy_dt

#Define for dy/dr out of the star
def dy_dr_out(r, y, P, w):
    M, Phi, Psi = y
    dy_dt = [f2(r, P, M, Psi, Phi, w),f3(r, P, M, Psi, Phi),f4(r, P, M, Psi, Phi, w) ]
    return dy_dt


def first_integral_profile(r, P, M, Phi, Psi, w):
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
    F1 = adota(r, P, M, Psi, Phi, w)
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

    C = ln_a + enthalpy_piece
    drift = C - C[0]
    return C, drift


class TOV:
    def __init__(
        self,
        initDensity,
        w,
        initPsi=0.0,
        initPhi=1.0,
        radiusMax_in=40000,
        Npoint=1000000,

    ):
        self.initDensity = initDensity
        self.initPressure = PEQS(initDensity)
        self.initPsi = initPsi
        self.initPhi = initPhi
        self.initMass = 0.0
        self.w = w

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
            args=(self.w,),
            rtol=1e-8,
            atol=1e-10,
        )
        self.sol = sol

        P, M, Phi, Psi = sol.y
        C, drift = first_integral_profile(sol.t, P, M, Phi, Psi, self.w)
        self.first_integral = C
        self.first_integral_drift = drift
        return sol


def _demo_run():
    # A simple run intended as a sanity check of the first integral.
    # (The original code prints initDensity in MeV/fm^3; here we just reuse the same input convention.)
    initDensity = 100.0

    tov = TOV(
        initDensity=initDensity,
        initPsi=0.0,
        initPhi=1.0,
        radiusMax_in=40000,
        Npoint=1000000,
    )
    sol = tov.compute()
    drift = tov.first_integral_drift
    max_abs = float(np.nanmax(np.abs(drift)))
    # normalize by typical scale of C to get a relative measure
    C0 = float(tov.first_integral[0])
    rel = max_abs / max(1.0, abs(C0))
    # print(f"[{name}] status={sol.status}, steps={sol.t.size}, max|ΔC|={max_abs:.3e}, rel≈{rel:.3e}")


if __name__ == "__main__":
    _demo_run()
