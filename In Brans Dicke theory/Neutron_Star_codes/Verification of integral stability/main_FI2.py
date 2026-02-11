#!/usr/bin/env python3
"""
main.py — Run the modified TOV (TOV_first_integral_check.py), report
(M_star, R_star), and plot the first-integral "constant" C(r) inside the star.

Place this file next to:
  - TOV_first_integral_check.py
then run:
  python3 main.py
"""

import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt

from TOV_first_integral_check import TOV, massSun

from matplotlib.transforms import Bbox
import mplhep as hep
hep.style.use("ATLAS")


def mevfm3_to_mass_density_SI(e_mevfm3: float) -> float:
    """Convert an energy density in MeV/fm^3 to mass density in kg/m^3.

    e [MeV/fm^3] -> e [J/m^3] via (MeV -> J) and (fm^3 -> m^3),
    then rho = e / c^2.
    """
    e_J_m3 = e_mevfm3 * 1e6 * cst.eV / (cst.fermi ** 3)
    rho_kg_m3 = e_J_m3 / (cst.c ** 2)
    return float(rho_kg_m3)


def find_surface_index(r: np.ndarray, P: np.ndarray, frac_floor: float = 1e-12) -> int:
    """Return index of the stellar surface based on pressure.

    We define the surface as the first radius where:
      P <= 0  OR  P <= frac_floor * P_c
    This is robust even if the integrator never hits exactly P=0.
    """
    P = np.asarray(P)
    r = np.asarray(r)
    Pc = float(P[0])
    floor = max(0.0, frac_floor * Pc)

    # If pressure becomes non-finite, stop before that
    bad = np.where(~np.isfinite(P))[0]
    if bad.size > 0:
        return max(0, int(bad[0] - 1))

    # First crossing of <=0
    idx = np.where(P <= 0.0)[0]
    if idx.size > 0:
        return int(idx[0])

    # Otherwise use floor fraction of central pressure
    idx2 = np.where(P <= floor)[0]
    if idx2.size > 0:
        return int(idx2[0])

    # If never decreases enough, return last point
    return int(len(r) - 1)


def run_case(init_density_kg_m3: float,w, radiusMax_in: float = 3.0e4, Npoint: int = 1000000):
    tov = TOV(
        initDensity=init_density_kg_m3,
        w=w,
        initPsi=0.0,
        initPhi=1.0,
        radiusMax_in=radiusMax_in,
        Npoint=Npoint,
    )
    sol = tov.compute()
    r = sol.t
    P, M, Phi, Psi = sol.y

    i_surf = find_surface_index(r, P)
    R_star_m = float(r[i_surf])
    M_star_kg = float(M[i_surf])

    # First integral constant profile inside the star
    C = np.asarray(tov.first_integral)[: i_surf + 1]
    r_in = np.asarray(r)[: i_surf + 1]

    drift = np.asarray(tov.first_integral_drift)[: i_surf + 1]
    max_abs = float(np.nanmax(np.abs(drift)))

    return {
        "status": sol.status,
        "n_steps": int(r.size),
        "R_star_m": R_star_m,
        "R_star_km": R_star_m / 1e3,
        "M_star_kg": M_star_kg,
        "M_star_solar": M_star_kg / massSun,
        "max_abs_drift": max_abs,
        "r_in_m": r_in,
        "r_in_km": r_in / 1e3,
        "C_in": C,
        "drift_in": drift,
    }


def main(w):
    # Choose your central density convention
    USE_MEVFM3 = True

    if USE_MEVFM3:
        e0 = 1500.0  # MeV/fm^3 (energy density)
        init_density = mevfm3_to_mass_density_SI(e0)
        print(f"Central energy density: {e0:.3g} MeV/fm^3 -> rho_c = {init_density:.6e} kg/m^3")
    else:
        init_density = 1500.0  # kg/m^3 (demo value, not astrophysically meaningful)
        print(f"Using initDensity = {init_density} kg/m^3 (demo/sanity-check value)")

    cases = [
        (1, "Lm = -rho c^2"),
    ]


    # cases = [
    #     (2, "Lm = P"),
    #     (1, "Lm = -rho c^2"),
    #     (0, "Lm = T"),
    # ]

    results = []
    for name in cases:
        res = run_case(init_density, w)
        results.append(res)
        #
        # print(
        #     f"[{name}] status={res['status']}  steps={res['n_steps']}  "
        #     f"R*={res['R_star_km']:.3f} km  M*={res['M_star_solar']:.6f} Msun  "
        #     f"max|ΔC|={res['max_abs_drift']:.3e}"
        # )

    c_in = []
    for res in results:
        c_in.append(res["C_in"])

    variation = []
    for i in range(len(c_in[0])):
        variation.append(c_in[0][i] - c_in[0][0])
    # print(variation)
    # Plot C(r) inside the star for each case
    plt.figure()
    for res in results:
        plt.plot(res["r_in_km"], variation)
    plt.xlabel("radius (km)")
    plt.ylabel(f"$\Delta C(r)$")
    # plt.ylabel(f"Variation of first integral constant  $\Delta C(r)$")
    # plt.title("First-integral consistency check inside the star")
    # plt.legend()
    plt.tight_layout()
    out_png = "first_integral_constant.png"
    plt.savefig(out_png, dpi=200)
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main(w = 10)
