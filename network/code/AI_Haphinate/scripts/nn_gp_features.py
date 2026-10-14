#!/usr/bin/env python3
"""Amendment 2, R3: Zeq and G-P buildup of the 40 development compositions on the shallow grid.

EPICS2017-based tables of each composition (validation/epics_results), coefficients interpolated linearly in ln E
as in the article (composition_screen.interp). Writes step4/gp_features.csv. Development compositions only.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from composition_screen import epics_table, gp_B, interp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation/highE_extension_20261003/surrogate_17E/training_data.csv"
OUT = ROOT / "validation/nn_tests_20261010/step4/gp_features.csv"


def main():
    keys = pd.read_csv(DATA)[["material_id", "energy_MeV", "depth_mfp"]]
    tabs = {(m, q): epics_table(m, q) for m in keys.material_id.unique() for q in ("EBF", "EABF")}
    rows = []
    for m, e, x in keys.itertuples(index=False):
        r = dict(material_id=m, energy_MeV=e, depth_mfp=x)
        for q in ("EBF", "EABF"):
            c = interp(tabs[m, q], e)
            r["Zeq_epics"] = float(c.Zeq) if q == "EBF" else r["Zeq_epics"]
            r["BGP_" + q] = gp_B(x, c.b, c.c, c.a, c.Xk, c.d)
        rows.append(r)
    out = pd.DataFrame(rows)
    assert len(out) == 40 * 17 * 8 and out.notna().all().all() and (out[["BGP_EBF", "BGP_EABF"]] > 1).all().all()
    out.to_csv(OUT, index=False)
    print(out.describe().T[["min", "max"]])


if __name__ == "__main__":
    main()
