#!/usr/bin/env python3
"""First-principles fluorescence buildup for a point isotropic source in an
infinite hafnate medium (no fitted parameters).

Units: optical depth tau = mu0 r, mu0 = total attenuation at E0 without
coherent scattering (BF/ANS-6.4.3 convention). Uncollided flux
phi0 = exp(-tau) / (4 pi tau^2) per source photon.

Generation 1: primary photons create line-l photons (all elements, K and L
lines, xraylib CS_FluorLine_Kissel_Cascade) at k_l = sigma_l(E0)/mu0 per unit
tau; a spherical shell source S(t) dt gives
    phi(tau) = int S(t) [E1(a|tau-t|) - E1(a(tau+t))] / (8 pi tau t) dt,
a = mu(E_l)/mu0, with S0(t) = k_l exp(-t).
Generation 2: line-l photons absorbed with fluorescence into line m
(sigma_m(E_l)/mu0 per unit tau), S1(t) = (sigma_m(E_l)/mu0) phi_l(t) 4 pi t^2.
Lines with a > 30 are treated as locally absorbed (phi/phi0 = k/a).
Fluorescence photons are not scattered (Compton is ~5 % of attenuation at
40-65 keV here), and electron-induced x-rays (PIXE) are ignored.

B_fluo = sum_lines phi_l E_l R(E_l) / (phi0 E0 R(E0)), R = mu_en of air (EBF)
or of the material (EABF), from the same tables BF uses.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import xraylib as xr
from scipy.integrate import quad
from scipy.special import exp1

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "validation" / "response_20260928" / "candidate_tables"
AIR = ROOT / "validation" / "response_20260928" / "reference_air.csv"
ELEMENTS = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Y", "Yb", "Lu", "Sc", "Hf", "O"]
# every K and L transition known to xraylib; M lines (< 4 keV) are left out
# (Geant4 shows almost no response below 5 keV in the fine spectra)
LINES = {n[:-5]: getattr(xr, n) for n in dir(xr) if re.fullmatch(r"(K|L[123])[K-Q]\d{1,2}_LINE", n)}
MIN_LINE_KEV = 4.0
LOCAL_A = 30.0


def table(path):
    import pandas as pd
    t = pd.read_csv(path)
    return t.iloc[:, 0].values, t.iloc[:, 1].values


def interp_loglog(tab, E):
    e, m = tab
    return float(np.exp(np.interp(math.log(E), np.log(e), np.log(m))))


class Material:
    def __init__(self, mid, weights):
        self.mid = mid
        self.w = {e: f for e, f in weights.items() if f > 0}
        self.Z = {e: xr.SymbolToAtomicNumber(e) for e in self.w}
        self.muen_mat = table(TABLES / f"{mid}.csv")
        self.muen_air = table(AIR)

    @lru_cache(maxsize=None)
    def mu(self, keV):
        """Mass attenuation without coherent scattering (cm2/g)."""
        def one(z):
            try:
                return xr.CS_Total_Kissel(z, keV) - xr.CS_Rayl(z, keV)
            except ValueError:  # Kissel photoionization data missing for this (Z, E)
                return xr.CS_Total(z, keV) - xr.CS_Rayl(z, keV)
        return sum(f * one(self.Z[e]) for e, f in self.w.items())

    @lru_cache(maxsize=None)
    def incoh(self, keV):
        return sum(f * xr.CS_Compt(self.Z[e], keV) for e, f in self.w.items())

    @lru_cache(maxsize=None)
    def lines(self, keV):
        """[(element, line, E_line keV, production cross section cm2/g of mixture)] for photons of energy keV."""
        out = []
        for e, f in self.w.items():
            for name, code in LINES.items():
                try:
                    s = xr.CS_FluorLine_Kissel_Cascade(self.Z[e], code, keV)
                    El = xr.LineEnergy(self.Z[e], code)
                except ValueError:
                    continue
                if s > 0 and El >= MIN_LINE_KEV:
                    out.append((e, name, El, f * s))
        return tuple(out)


def shell_kernel(tau, t, a):
    return (exp1(a * abs(tau - t)) - exp1(a * (tau + t))) / (8 * math.pi * tau * t)


def gen1_ratio(k, a, tau):
    """phi_line / phi0 for S0(t) = k exp(-t)."""
    if a > LOCAL_A:
        return k / a
    f = lambda t: k * math.exp(-t) * shell_kernel(tau, t, a)
    hi = tau + 40.0 / min(a, 1.0) + 40.0
    val = quad(f, 0, tau, limit=200)[0] + quad(f, tau, hi, limit=400)[0]
    return val * 4 * math.pi * tau ** 2 * math.exp(tau)


def buildup_fluo(m: Material, E0_MeV, depths, generations=2):
    E0 = 1000 * E0_MeV
    mu0 = m.mu(E0)
    R_air0 = E0 * interp_loglog(m.muen_air, E0_MeV)
    R_mat0 = E0 * interp_loglog(m.muen_mat, E0_MeV)
    res = {x: dict(EBF=0.0, EABF=0.0, EBF_g2=0.0, EABF_g2=0.0) for x in depths}
    grid = np.concatenate([np.linspace(1e-3, 2, 120), np.linspace(2.02, 60, 400)])
    for (e, name, El, s) in m.lines(E0):
        k, a = s / mu0, m.mu(El) / mu0
        wa, wm = El * interp_loglog(m.muen_air, El / 1000), El * interp_loglog(m.muen_mat, El / 1000)
        for x in depths:
            g1 = gen1_ratio(k, a, x)
            res[x]["EBF"] += g1 * wa / R_air0
            res[x]["EABF"] += g1 * wm / R_mat0
        if generations < 2 or a > LOCAL_A:
            continue
        # generation 2: this line's photons re-absorbed with fluorescence
        sec = [(e2, n2, E2, s2) for (e2, n2, E2, s2) in m.lines(El) if s2 > 0]
        if not sec:
            continue
        phi1 = np.array([gen1_ratio(k, a, t) * math.exp(-t) / (4 * math.pi * t ** 2) for t in grid])  # absolute phi_l(t)
        S1_base = phi1 * 4 * math.pi * grid ** 2  # times sigma_m(E_l)/mu0 below
        for (e2, n2, E2, s2) in sec:
            k2, a2 = s2 / mu0, m.mu(E2) / mu0
            wa2, wm2 = E2 * interp_loglog(m.muen_air, E2 / 1000), E2 * interp_loglog(m.muen_mat, E2 / 1000)
            for x in depths:
                if a2 > LOCAL_A:
                    g2 = k2 * np.interp(x, grid, S1_base) / (4 * math.pi * x ** 2) / a2 / (math.exp(-x) / (4 * math.pi * x ** 2))
                else:
                    f = lambda t: k2 * np.interp(t, grid, S1_base) * shell_kernel(x, t, a2)
                    g2 = (quad(f, grid[0], x, limit=200)[0] + quad(f, x, grid[-1], limit=400)[0]) \
                        * 4 * math.pi * x ** 2 * math.exp(x)
                res[x]["EBF"] += g2 * wa2 / R_air0
                res[x]["EABF"] += g2 * wm2 / R_mat0
                res[x]["EBF_g2"] += g2 * wa2 / R_air0
                res[x]["EABF_g2"] += g2 * wm2 / R_mat0
    return res


# ---------------------------------------------------------------------------
# Full cascade (added 2026-09-28): every generation of line photons is a source
# for the next, until a generation adds < tol to the buildup. Works with the
# reduced track density T(t) = 4 pi t^2 phi(t) (primary: T0 = exp(-t)); a line
# produced at S(t) per unit tau gives
#   T(tau) = int S(t) tau/(2t) [E1(a|tau-t|) - E1(a(tau+t))] dt,
# or T = S/a for locally absorbed lines (a > LOCAL_A).
def _grid(depths, refinement=0, lower_radius=1e-3):
    if not isinstance(refinement, (int, np.integer)) or refinement < 0:
        raise ValueError('radial refinement must be a nonnegative integer')
    if not np.isfinite(lower_radius) or not 0 < lower_radius <= 1e-3:
        raise ValueError('lower radial cutoff must be positive and at most 0.001 MFP')
    top = max(30.0, max(depths) + 20.0)
    g = np.unique(np.concatenate([np.geomspace(1e-3, 0.2, 12), np.linspace(0.25, 6, 47),
                                  np.linspace(6.5, top, int(2 * (top - 6)) + 1), np.array(depths, float)]))
    if lower_radius < g[0]:
        cells = int(np.ceil(8*np.log10(g[0]/lower_radius)))
        g = np.r_[np.geomspace(lower_radius, g[0], cells+1)[:-1], g]
    for _ in range(refinement):
        g = np.sort(np.r_[g, (g[:-1]+g[1:])/2])
    return g


def _propagate_legacy(grid, S, a):
    """Original adaptive routine, retained to reproduce archived predictions."""
    out = np.empty_like(grid)
    f_S = lambda t: np.interp(t, grid, S)
    for i, tau in enumerate(grid):
        f = lambda t: f_S(t) * tau / (2 * t) * (exp1(a * abs(tau - t)) - exp1(a * (tau + t)))
        out[i] = quad(f, grid[0], tau, limit=100, epsrel=1e-4)[0] + quad(f, tau, grid[-1], limit=100, epsrel=1e-4)[0]
    return out


def _propagate(grid, S, a):
    """Integrate interpolation cells explicitly, including the log singularity.

    The old default absolute tolerance can dominate deep, tiny track densities.
    Cell quadrature is linear in the source and independent of its absolute scale.
    Spatial interpolation error requires its own grid/cutoff convergence check.
    """
    from radial_quadrature import propagate
    return propagate(grid, S, a, regular_order=16, singular_order=128)


# Scattered source photons (fluence ratio to uncollided) at 0.1 MeV, mean of the
# 15 no-fluorescence Geant4 runs; used as an optional extra fluorescence source.
SCATTER_SOURCE_0P1 = ((0.0, 0.0), (0.5, 0.021), (1.0, 0.030), (2.0, 0.041), (3.0, 0.047), (4.0, 0.057))


def buildup_fluo_cascade(m: Material, E0_MeV, depths, max_gen=6, tol=1e-4, scatter_source=None,
                         straight_ahead=False, radial_refinement=0, radial_min=1e-3,
                         integration="cell"):
    """scatter_source: ((depth, ratio), ...) adds scattered source photons (at E0) to the
    generation-0 source; straight_ahead: incoherent scattering of line photons is
    treated as non-removal (upper bound on their scatter buildup).
    radial_refinement bisects interpolation cells; radial_min is the lower cutoff
    in MFP. integration='legacy' reproduces the archived adaptive integrator.
    Default grid settings are the original grid; they do not certify spatial
    convergence. See research_notes/radial_correction_20260930 for the audit.
    """
    if integration not in ("cell", "legacy"):
        raise ValueError("integration must be 'cell' or 'legacy'")
    propagator = _propagate if integration == "cell" else _propagate_legacy
    E0 = 1000 * E0_MeV
    mu0 = m.mu(E0)
    R_air0 = E0 * interp_loglog(m.muen_air, E0_MeV)
    R_mat0 = E0 * interp_loglog(m.muen_mat, E0_MeV)
    grid = _grid(depths, radial_refinement, radial_min)
    idx = [int(np.argmin(abs(grid - x))) for x in depths]
    T0 = np.exp(-grid)
    T_src = T0 if scatter_source is None else T0 * (1 + np.interp(grid, *zip(*scatter_source)))
    res = {x: dict(EBF=0.0, EABF=0.0, **{f"EBF_gen{g}": 0.0 for g in range(1, max_gen + 1)}) for x in depths}
    parents = [(E0, T_src)]
    for gen in range(1, max_gen + 1):
        sources = {}
        for Ep, Tp in parents:
            for (e, name, El, s) in m.lines(Ep):
                key = (e, name, round(El, 4))
                sources[key] = sources.get(key, 0.0) + (s / mu0) * Tp
        if not sources:
            break
        added, parents = 0.0, []
        for (e, name, El), S in sources.items():
            a = m.mu(El) / mu0
            if straight_ahead and a <= LOCAL_A:
                a = (m.mu(El) - m.incoh(El)) / mu0
            T = S / a if a > LOCAL_A else propagator(grid, S, a)
            parents.append((El, T))
            wa, wm = El * interp_loglog(m.muen_air, El / 1000), El * interp_loglog(m.muen_mat, El / 1000)
            for x, i in zip(depths, idx):
                r = T[i] / T0[i]
                res[x]["EBF"] += r * wa / R_air0
                res[x]["EABF"] += r * wm / R_mat0
                res[x][f"EBF_gen{gen}"] += r * wa / R_air0
                added = max(added, r * wa / R_air0)
        if added < tol:
            break
    return res
