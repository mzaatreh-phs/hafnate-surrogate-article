#!/usr/bin/env python3
"""XCOM-based inputs of the published-style sets for the prospective K-edge cohort (system python3, epixs_cli).

Same functions as scripts/nn_heldout_xcom.py (Amendment 5a of validation/nn_tests_20261010). No label file is read.
Reads validation/prospective_kedge_20261011/frozen/inputs.csv and writes frozen/xcom_features.csv.
"""
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nn_heldout_xcom import EL, N, mass_attenuation, zeq  # noqa: E402

FZ = Path(__file__).resolve().parents[1] / "validation/prospective_kedge_20261011/frozen"


def main():
    edges = {r["element"]: float(r["EK_MeV"]) for r in csv.DictReader((N / "step4/k_edges.csv").open())}
    rows = list(csv.DictReader((FZ / "inputs.csv").open()))
    cache, out = {}, []
    for r in rows:
        m, E, d, rho = r["material_id"], float(r["energy_MeV"]), float(r["depth_mfp"]), float(r["density_g_cm3"])
        if (m, E) not in cache:
            comp = {e: float(r[e + "_wt"]) for e in EL if float(r[e + "_wt"]) > 0}
            x = mass_attenuation(comp, [E])
            cat = {e: w for e, w in comp.items() if e != "O"}
            tot = sum(cat.values())
            cache[m, E] = dict(Zeq_xcom=zeq(comp, E), inc=x["incoherent"][0], pe=x["photoelectric"][0], tot=x["total_without_coherent"][0],
                               mean=sum(w * edges[e] for e, w in cat.items()) / tot, min=min(edges[e] for e in cat), max=max(edges[e] for e in cat))
        c = cache[m, E]
        o = dict(material_id=m, energy_MeV=r["energy_MeV"], depth_mfp=r["depth_mfp"], Zeq_xcom=c["Zeq_xcom"],
                 log_mu=math.log(c["tot"]), log_S_inc=math.log(c["inc"] * rho), log_S_pe=math.log(c["pe"] * rho), log_S_tot=math.log(c["tot"] * rho))
        for k in ("mean", "min", "max"):
            o["EK_" + k] = c[k]
            o["mek_" + k] = d / (E - c[k])
        out.append(o)
    assert all(o["Zeq_xcom"] is not None for o in out)
    with (FZ / "xcom_features.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(len(out), "rows")


if __name__ == "__main__":
    main()
