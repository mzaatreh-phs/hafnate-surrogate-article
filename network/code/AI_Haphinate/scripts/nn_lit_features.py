#!/usr/bin/env python3
"""Amendment 3, R5: published-style input sets for the 40 development compositions.

XCOM partial cross sections through epixs_cli (mixture rule on the mass fractions) and K-edge energies written by the
caller. Run with system python3 (epixs_cli's own environment); K-edge energies come from --edges, a CSV written with
xraylib in the project environment. Writes step4/lit_features.csv.
"""
import csv
import math
import sys
from pathlib import Path

EPIXS = Path.home() / "epixs_cli"
sys.path[:0] = [str(EPIXS), str(EPIXS / "epixs")]
from epixs.xcom_engine import mass_attenuation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N = ROOT / "validation/nn_tests_20261010/step4"
DATA = ROOT / "validation/highE_extension_20261003/surrogate_17E/training_data.csv"
EL = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Y", "Yb", "Lu", "Sc", "Hf", "O"]


def main():
    edges = {r["element"]: float(r["EK_MeV"]) for r in csv.DictReader((N / "k_edges.csv").open())}
    ek_z = {int(r["Z"]): float(r["EK_MeV"]) for r in csv.DictReader((N / "k_edges_by_Z.csv").open())}
    zeq = {(r["material_id"], r["energy_MeV"], r["depth_mfp"]): float(r["Zeq_epics"])
           for r in csv.DictReader((N / "gp_features.csv").open())}
    rows = list(csv.DictReader(DATA.open()))
    mats = {}
    for r in rows:
        mats.setdefault(r["material_id"], r)
    energies = sorted({float(r["energy_MeV"]) for r in rows})
    xs = {}
    for m, r in mats.items():
        comp = {e: float(r[e + "_wt"]) for e in EL if float(r[e + "_wt"]) > 0}
        x = mass_attenuation(comp, energies)
        cat = {e: w for e, w in comp.items() if e != "O"}
        tot = sum(cat.values())
        ek = dict(mean=sum(w * edges[e] for e, w in cat.items()) / tot, min=min(edges[e] for e in cat),
                  max=max(edges[e] for e in cat))
        for i, E in enumerate(x["energy_MeV"]):
            xs[m, round(E, 6)] = dict(inc=x["incoherent"][i], pe=x["photoelectric"][i], tot=x["total_without_coherent"][i], **ek)
    out = []
    for r in rows:
        m, E, d, rho = r["material_id"], float(r["energy_MeV"]), float(r["depth_mfp"]), float(r["density_g_cm3"])
        c = xs[m, round(E, 6)]
        z = zeq[m, r["energy_MeV"], r["depth_mfp"]]
        lo = int(math.floor(z))
        ekz = ek_z[lo] + (z - lo) * (ek_z[lo + 1] - ek_z[lo])
        o = dict(material_id=m, energy_MeV=r["energy_MeV"], depth_mfp=r["depth_mfp"], log_mu=math.log(c["tot"]),
                 EK_zeq=ekz, mek_zeq=d / (E - ekz), log_S_inc=math.log(c["inc"] * rho), log_S_pe=math.log(c["pe"] * rho),
                 log_S_tot=math.log(c["tot"] * rho))
        for k in ("mean", "min", "max"):
            o["EK_" + k] = c[k]
            o["mek_" + k] = d / (E - c[k])
        out.append(o)
    assert len(out) == 40 * 17 * 8
    with (N / "lit_features.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(len(out), "rows", {k: (min(o[k] for o in out), max(o[k] for o in out)) for k in ("EK_zeq", "EK_mean", "EK_min", "EK_max", "mek_zeq", "log_mu")})


if __name__ == "__main__":
    main()
