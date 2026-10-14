#!/usr/bin/env python3
"""Composition screen (2026-09-28): is there a composition effect on buildup
beyond Monte Carlo noise and beyond Zeq, and where does G-P depart from Geant4?

Labels: 40 development materials + HEH1 (ml/buildup/buildup_dataset.csv) + 5
confirmatory materials (~/BF/runs/labels_confirm_20260927), at the 13 grid
energies x 8 grid depths (0.5-4 MFP).

G-P reference: EPICS2017 (EpiXS) G-P coefficient files in
validation/epics_results/<id>_{EBF,EABF}_EPICS2017.csv (ANS-6.4.3 1991
coefficients interpolated to Zeq(E)); off-grid energies (0.25, 0.6617, 1.25 MeV)
by log-log interpolation of the coefficients between tabulated energies, as
EpiXS does. The 5 confirmatory materials have no EPICS2017 file; they take
Zeq and G-P from epixs_cli (validation/composition_screen_20260928/gp_epixs.csv)
except at 0.1 MeV, where epixs_cli clamps Zeq to 92 (non-monotonic
Compton/total ratio below the U K-edge) and they are left out.

1. Spread vs noise: between-material SD of ln B, and the SD left after a
   quadratic fit of ln B on Zeq, against the label noise; "excess" =
   sqrt(max(0, residual variance - mean label variance)).
2. Geant4 / G-P ratio by energy and depth.
3. Matched-Zeq pairs (|dZeq| <= 0.25): Geant4 difference and its z-score.
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
OUT = ROOT / "validation" / "composition_screen_20260928"
EPICS = ROOT / "validation" / "epics_results"
CONFIRM = Path.home() / "BF" / "runs" / "labels_confirm_20260927" / "labels.csv"
DEPTHS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
TARGETS = ("EBF", "EABF")


def gp_B(x, b, c, a, Xk, d):
    K = c * x ** a + d * (math.tanh(x / Xk - 2) - math.tanh(-2)) / (1 - math.tanh(-2))
    if abs(K - 1) < 1e-9:
        return 1 + (b - 1) * x
    return 1 + (b - 1) * (K ** x - 1) / (K - 1)


def epics_table(mid, q):
    t = pd.read_csv(EPICS / f"{mid}_{q}_EPICS2017.csv", encoding="utf-8-sig")
    t = t[t["Grid"] == "Standard"]
    t = t.rename(columns={t.columns[1]: "keV", "Xₖ": "Xk"})
    t["E"] = t.keV.astype(float) / 1000
    t = t.set_index("E")[["Zeq", "b", "c", "a", "Xk", "d"]].apply(pd.to_numeric, errors="coerce")
    return t.dropna()  # a few files lack coefficients at 15 keV only


def interp(tab, E):
    Es = tab.index.values
    if np.any(np.isclose(Es, E)):
        return tab.loc[Es[np.isclose(Es, E)][0]]
    lo, hi = Es[Es < E].max(), Es[Es > E].min()
    f = (math.log(E) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return tab.loc[lo] + (tab.loc[hi] - tab.loc[lo]) * f


def load_labels():
    df = pd.read_csv(ROOT / "ml" / "buildup" / "buildup_dataset.csv")
    cf = pd.read_csv(CONFIRM)
    lab = pd.concat([df[["material_id", "energy_MeV", "depth_mfp", "EBF", "EBF_se", "EABF", "EABF_se"]],
                     cf[["material_id", "energy_MeV", "depth_mfp", "EBF", "EBF_se", "EABF", "EABF_se"]]])
    lab = lab[lab.depth_mfp.round(6).isin(DEPTHS) & ~np.isclose(lab.energy_MeV, 0.122474)]
    # HEH1 at 0.6617/1.25/2.0 comes from the pilot (15 depths); keep grid depths only
    lab = lab.groupby(["material_id", "energy_MeV", "depth_mfp"], as_index=False).first()
    return lab


def add_gp(lab):
    out = []
    epixs = pd.read_csv(OUT / "gp_epixs.csv")
    for mid, g in lab.groupby("material_id"):
        if (EPICS / f"{mid}_EBF_EPICS2017.csv").exists():
            tabs = {q: epics_table(mid, q) for q in TARGETS}
            for _, r in g.iterrows():
                row = r.to_dict()
                for q in TARGETS:
                    c = interp(tabs[q], r.energy_MeV)
                    row[q + "_GP"] = gp_B(r.depth_mfp, c.b, c.c, c.a, c.Xk, c.d)
                row["Zeq"] = interp(tabs["EBF"], r.energy_MeV).Zeq
                row["gp_source"] = "EPICS2017"
                out.append(row)
        else:
            e = epixs[epixs.material_id == mid]
            for _, r in g.iterrows():
                row = r.to_dict()
                m = e[np.isclose(e.energy_MeV, r.energy_MeV) & np.isclose(e.depth_mfp, r.depth_mfp)].iloc[0]
                ok = r.energy_MeV > 0.11
                row.update(EBF_GP=m.EBF_GP if ok else np.nan, EABF_GP=m.EABF_GP if ok else np.nan,
                           Zeq=m.Zeq if ok else np.nan, gp_source="epixs_cli")
                out.append(row)
    return pd.DataFrame(out)


def validate_gp():
    """Our G-P formula vs the EPICS2017 file's own tabulated EBF/EABF at 1-4 MFP."""
    worst = 0.0
    for f in EPICS.glob("*_EBF_EPICS2017.csv"):
        mid = f.name.split("_EBF_")[0]
        for q in TARGETS:
            t = pd.read_csv(EPICS / f"{mid}_{q}_EPICS2017.csv", encoding="utf-8-sig")
            t = t[t["Grid"] == "Standard"].rename(columns={"Xₖ": "Xk"})
            for _, r in t.iterrows():
                for x in (1, 2, 3, 4):
                    if str(r[f"{q} at {x}-MFP"]).strip() in ("-", "", "nan"):
                        continue
                    ref = float(r[f"{q} at {x}-MFP"])
                    worst = max(worst, abs(gp_B(x, *(float(r[k]) for k in ("b", "c", "a", "Xk", "d"))) / ref - 1))
    return worst


def spread(d):
    rows = []
    for (E, x), g in d.groupby(["energy_MeV", "depth_mfp"]):
        for q in TARGETS:
            y = np.log(g[q].values)
            noise_var = np.mean((g[q + "_se"].values / g[q].values) ** 2)
            h = g.dropna(subset=["Zeq"])
            yz = np.log(h[q].values)
            coef = np.polyfit(h.Zeq.values, yz, 2)
            res = yz - np.polyval(coef, h.Zeq.values)
            rv = res.var(ddof=3)
            rows.append(dict(target=q, energy_MeV=E, depth_mfp=x, n=len(g), sd_all=100 * y.std(ddof=1),
                             sd_after_Zeq=100 * math.sqrt(rv), noise=100 * math.sqrt(noise_var),
                             excess_beyond_Zeq=100 * math.sqrt(max(0.0, rv - np.mean((h[q + "_se"] / h[q]) ** 2))),
                             chi2_ratio=rv / np.mean((h[q + "_se"] / h[q]) ** 2)))
    return pd.DataFrame(rows)


def pairs(d, tol=0.25):
    rows = []
    for E, g in d.dropna(subset=["Zeq"]).groupby("energy_MeV"):
        z = g.groupby("material_id").Zeq.first()
        for a, b in itertools.combinations(sorted(z.index), 2):
            if abs(z[a] - z[b]) > tol:
                continue
            ga = g[g.material_id == a].set_index("depth_mfp")
            gb = g[g.material_id == b].set_index("depth_mfp")
            for q in TARGETS:
                diff = np.log(ga[q] / gb[q])
                se = np.sqrt((ga[q + "_se"] / ga[q]) ** 2 + (gb[q + "_se"] / gb[q]) ** 2)
                zz = diff / se
                rows.append(dict(target=q, energy_MeV=E, a=a, b=b, Zeq_a=z[a], Zeq_b=z[b],
                                 diff_at_4mfp_pct=100 * diff.loc[4.0], z_at_4mfp=zz.loc[4.0],
                                 max_abs_diff_pct=100 * diff.abs().max(), chi2_8=float((zz ** 2).sum()),
                                 gp_diff_pct=100 * math.log(ga[q + "_GP"].loc[4.0] / gb[q + "_GP"].loc[4.0])))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    worst = validate_gp()
    d = add_gp(load_labels())
    for q in TARGETS:
        d[q + "_ratio_pct"] = 100 * (d[q] / d[q + "_GP"] - 1)
    d.to_csv(OUT / "labels_with_gp.csv", index=False)
    sp = spread(d)
    sp.to_csv(OUT / "spread_vs_noise.csv", index=False)
    pr = pairs(d)
    pr.to_csv(OUT / "matched_zeq_pairs.csv", index=False)

    ep = pd.read_csv(OUT / "gp_epixs.csv")
    chk = d[d.gp_source == "EPICS2017"].merge(ep, on=["material_id", "energy_MeV", "depth_mfp"], suffixes=("", "_epx"))
    agree = chk[chk.energy_MeV > 0.11].assign(dz=lambda t: (t.Zeq - t.Zeq_epx).abs(),
                                               de=lambda t: 100 * (t.EBF_GP_epx / t.EBF_GP - 1).abs())

    L = ["# Composition screen (2026-09-28)", "", "`scripts/composition_screen.py`; design in its docstring.", "",
         f"- G-P formula reproduces the EPICS2017 files' own tabulated values to {100 * worst:.3f} % (1-4 MFP, all energies).",
         f"- epixs_cli vs EPICS2017 for the 41 materials above 0.1 MeV: max |dZeq| {agree.dz.max():.2f}, "
         f"max EBF difference {agree.de.max():.2f} %. At 0.1 MeV epixs_cli returns Zeq = 92 (clamped) against "
         f"EPICS2017 {d[(d.gp_source == 'EPICS2017') & np.isclose(d.energy_MeV, 0.1)].Zeq.mean():.1f}; not used there.",
         f"- Materials: {d.material_id.nunique()}; rows {len(d)}.", ""]
    L += ["## 1. Composition effect vs Monte Carlo noise (per energy, averaged over the 8 depths; % in ln B)", "",
          "| Target | E (MeV) | SD across materials | SD after Zeq fit | label noise | excess beyond Zeq | resid. var / noise var | at 4 MFP: SD after Zeq / noise |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (q, E), g in sp.groupby(["target", "energy_MeV"], sort=False):
        g4 = g[g.depth_mfp == 4.0].iloc[0]
        L.append(f"| {q} | {E:g} | {g.sd_all.mean():.2f} | {g.sd_after_Zeq.mean():.2f} | {g.noise.mean():.2f} | "
                 f"{g.excess_beyond_Zeq.mean():.2f} | {g.chi2_ratio.mean():.1f} | {g4.sd_after_Zeq:.2f} / {g4.noise:.2f} |")
    L += ["", "## 2. Geant4 / G-P - 1 (%), mean over materials [min, max]", "",
          "| E (MeV) | EBF 1 MFP | EBF 4 MFP | EABF 1 MFP | EABF 4 MFP |", "|---:|---:|---:|---:|---:|"]
    for E, g in d.groupby("energy_MeV"):
        cells = []
        for q in TARGETS:
            for x in (1.0, 4.0):
                v = g[np.isclose(g.depth_mfp, x)][q + "_ratio_pct"].dropna()
                cells.append(f"{v.mean():+.1f} [{v.min():+.1f}, {v.max():+.1f}]")
        L.append(f"| {E:g} | " + " | ".join(cells) + " |")
    L += ["", "## 3. Matched-Zeq pairs (|dZeq| <= 0.25; G-P treats each pair as identical)", "",
          f"Pairs x energies x targets: {len(pr)}; with |z| > 3 at 4 MFP: {(pr.z_at_4mfp.abs() > 3).sum()}; "
          f"8-depth chi2 > 26.1 (p < 0.001): {(pr.chi2_8 > 26.1).sum()}.", "",
          "Largest Geant4 differences at 4 MFP:", "",
          "| Target | E (MeV) | pair | Zeq | Geant4 diff | z | G-P diff |", "|---|---:|---|---|---:|---:|---:|"]
    for _, r in pr.reindex(pr.diff_at_4mfp_pct.abs().sort_values(ascending=False).index).head(20).iterrows():
        L.append(f"| {r.target} | {r.energy_MeV:g} | {r.a} / {r.b} | {r.Zeq_a:.2f} / {r.Zeq_b:.2f} | "
                 f"{r.diff_at_4mfp_pct:+.1f} % | {r.z_at_4mfp:+.1f} | {r.gp_diff_pct:+.2f} % |")
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
