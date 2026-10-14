#!/usr/bin/env python3
"""Step 6 (AMENDMENT5.md, AMENDMENT5A.md): label-free input rows of the held-out cohorts.

Compositions, densities, the 100 keV reference and the atomic descriptors come from the files frozen by the earlier
evaluations of each cohort; no label file is read. Writes step6/heldout_inputs.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nn_mlp_lomo import ATOMIC, DATA, WT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "validation"
FR = V / "physics_surrogate_full_range_20260930"
OUT = V / "nn_tests_20261010/step6/heldout_inputs.csv"
E17 = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.6617, 0.8, 1.0, 1.25, 2.0, 3.0, 5.0, 6.0, 8.0, 10.0, 15.0]
D8 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
COLS = ["cohort", "material_id", "energy_MeV", "depth_mfp", "log10_E_MeV", "density_g_cm3"] + WT + ["anchor_EBF", "anchor_EABF"] + ATOMIC


def catalogue(mids, energies, depths, cohort):
    cat = pd.read_csv(ROOT / "materials/materials.csv").set_index("material_id")
    phys = pd.read_csv(FR / "retrospective_existing/physical_features.csv")
    rows = []
    for m in mids:
        for e in energies:
            for x in depths:
                r = dict(cohort=cohort, material_id=m, energy_MeV=e, depth_mfp=x, log10_E_MeV=np.log10(e),
                         density_g_cm3=cat.loc[m, "density_g_cm3"])
                r.update({w: cat.loc[m, w + "_fraction"] for w in WT})
                rows.append(r)
    return pd.DataFrame(rows).merge(phys, on=["material_id", "depth_mfp"], validate="many_to_one")


def frozen(path, cohort, energies=None):
    d = pd.read_csv(path).assign(cohort=cohort)
    return d if energies is None else d[d.energy_MeV.round(6).isin(energies)]


def main():
    cat = pd.read_csv(ROOT / "materials/materials.csv")
    kt = sorted(m for m in cat.material_id if m.startswith("KT_"))
    cf = sorted(m for m in cat.material_id if m.startswith("CF_"))
    assert len(kt) == 12 and len(cf) == 5
    parts = [catalogue(kt, [0.1, 0.15], [0.5, 1.0, 2.0, 3.0, 4.0], "kedge"),
             catalogue(cf, E17[:13], D8, "confirmatory"),
             catalogue(["HEH1"], E17, D8, "HEH1"),
             frozen(FR / "fresh_confirmation/frozen/prediction_inputs.csv", "fresh"),
             frozen(FR / "fresh_confirmation_amendment1/new6/frozen/prediction_inputs.csv", "fresh"),
             frozen(V / "blind_esgyp_20261008/frozen/grid_inputs.csv", "ESGYP"),
             frozen(V / "blind_esgyp_20261008/amendment1/frozen_lnttl/grid_inputs.csv", "LNTTL", [0.1, 0.15, 1.0, 15.0])]
    out = pd.concat([p[COLS] for p in parts], ignore_index=True)
    n = out.groupby("cohort").agg(materials=("material_id", "nunique"), rows=("material_id", "size"))
    print(n)
    assert dict(n.rows) == dict(kedge=120, confirmatory=520, HEH1=136, fresh=1248, ESGYP=136, LNTTL=32)
    assert out[COLS[4:]].notna().all().all() and not out.duplicated(["material_id", "energy_MeV", "depth_mfp"]).any()
    dev = pd.read_csv(DATA)
    assert not set(out.material_id) & set(dev.material_id)
    assert np.allclose(out[WT].sum(1), 1, atol=2e-4)
    # same catalogue conventions as the development table: check on development compositions
    c = cat.set_index("material_id"); d1 = dev.drop_duplicates("material_id").set_index("material_id")
    assert np.allclose(c.loc[d1.index, "density_g_cm3"], d1.density_g_cm3) and np.allclose(c.loc[d1.index, [w + "_fraction" for w in WT]].values, d1[WT].values)
    r = (out[["density_g_cm3", "anchor_EBF"] + ATOMIC].agg(["min", "max"]).T).join(dev[["density_g_cm3", "anchor_EBF"] + ATOMIC].agg(["min", "max"]).T, rsuffix="_dev")
    print(r.round(4))
    out.to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
