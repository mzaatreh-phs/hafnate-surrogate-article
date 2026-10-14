#!/usr/bin/env python3
"""Step 6 (AMENDMENT5A.md): XCOM-based inputs of the published-style sets for development and held-out rows.

Zeq as in scripts/zeq_table_xcom.py (same function), partial cross sections and K-edge inputs as in
scripts/nn_lit_features.py. Run with system python3 (epixs_cli environment). No label file is read.
Writes step6/xcom_features.csv (development rows from the training table keys, held-out rows from heldout_inputs.csv).
"""
import csv
import math
import sys
from functools import lru_cache
from pathlib import Path

EPIXS = Path.home() / "epixs_cli"
sys.path[:0] = [str(EPIXS), str(EPIXS / "epixs")]
from epixs.xcom_engine import mass_attenuation  # noqa: E402
from epixs.zeff import compton_to_total_ratio, element_compton_to_total_ratio, _get_xcom_engine  # noqa: E402
from atomic_data import SYMBOL_TO_Z  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N = ROOT / "validation/nn_tests_20261010"
DATA = ROOT / "validation/highE_extension_20261003/surrogate_17E/training_data.csv"
EL = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Y", "Yb", "Lu", "Sc", "Hf", "O"]
eng = _get_xcom_engine()


@lru_cache(maxsize=None)
def rz(z, e):
    return element_compton_to_total_ratio(z, e, eng)


def zeq(w, e):
    r = compton_to_total_ratio(w, e, eng)
    heavy = {k: v for k, v in w.items() if k != "O"}
    zref = sum(v * SYMBOL_TO_Z[k] for k, v in heavy.items()) / sum(heavy.values())
    best = None
    for z1 in range(1, 92):
        r1, r2 = rz(z1, e), rz(z1 + 1, e)
        if (r1 - r) * (r2 - r) <= 0 and r1 != r2:
            z = (z1 * (math.log10(r2) - math.log10(r)) + (z1 + 1) * (math.log10(r) - math.log10(r1))) / (math.log10(r2) - math.log10(r1))
            if best is None or abs(z - zref) < abs(best - zref):
                best = z
    return best


def main():
    edges = {r["element"]: float(r["EK_MeV"]) for r in csv.DictReader((N / "step4/k_edges.csv").open())}
    rows = [dict(r, part="development") for r in csv.DictReader(DATA.open())]
    rows += [dict(r, part="heldout") for r in csv.DictReader((N / "step6/heldout_inputs.csv").open())]
    mats = {}
    for r in rows:
        mats.setdefault(r["material_id"], r)
    cache = {}
    out = []
    for r in rows:
        m, E, d, rho = r["material_id"], float(r["energy_MeV"]), float(r["depth_mfp"]), float(r["density_g_cm3"])
        if (m, E) not in cache:
            comp = {e: float(mats[m][e + "_wt"]) for e in EL if float(mats[m][e + "_wt"]) > 0}
            x = mass_attenuation(comp, [E])
            cat = {e: w for e, w in comp.items() if e != "O"}
            tot = sum(cat.values())
            cache[m, E] = dict(Zeq_xcom=zeq(comp, E), inc=x["incoherent"][0], pe=x["photoelectric"][0], tot=x["total_without_coherent"][0],
                               mean=sum(w * edges[e] for e, w in cat.items()) / tot, min=min(edges[e] for e in cat), max=max(edges[e] for e in cat))
        c = cache[m, E]
        o = dict(part=r["part"], material_id=m, energy_MeV=r["energy_MeV"], depth_mfp=r["depth_mfp"], Zeq_xcom=c["Zeq_xcom"],
                 log_mu=math.log(c["tot"]), log_S_inc=math.log(c["inc"] * rho), log_S_pe=math.log(c["pe"] * rho), log_S_tot=math.log(c["tot"] * rho))
        for k in ("mean", "min", "max"):
            o["EK_" + k] = c[k]
            o["mek_" + k] = d / (E - c[k])
        out.append(o)
    assert all(o["Zeq_xcom"] is not None for o in out)
    with (N / "step6/xcom_features.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(len(out), "rows")


if __name__ == "__main__":
    main()
