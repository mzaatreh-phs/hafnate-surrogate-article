#!/usr/bin/env python3
"""Metrics for step 4 of validation/nn_tests_20261010 (AMENDMENT2.md, runs R1-R3).

Pools the step 3 networks, the step 4 networks, XGBoost tuned in the nested folds (tXGB-*), the article's
fixed-setting baselines, and the unfitted G-P and fluorescence-reference predictors. Sets ending in "d" use the
correction target ln(B / B_GP). Runs on whatever folds exist; incomplete methods are listed and left out.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nn_mlp_analyze as s3  # noqa: E402
from nn_mlp_lomo import load as load_data  # noqa: E402

N = Path(__file__).resolve().parents[1] / "validation/nn_tests_20261010"
PAIRS = [  # (a, b), AMENDMENT2.md
    ("MLP-DR", "MLP-Y2"), ("MLP-DR", "MLP-K1r"), ("MLP-PAy", "MLP-Y2"), ("MLP-PAy", "MLP-K1r"), ("MLP-DR", "MLP-PAy"),  # Amendment 4b, exploratory
    ("MLP-DR", "MLP-D"), ("MLP-P1", "MLP-PA"), ("MLP-P0", "MLP-DR"), ("XGBoost-F", "XGBoost-A"),  # R1
    ("tXGB-DR", "tXGB-D"), ("tXGB-P1", "tXGB-PA"), ("tXGB-P0", "tXGB-DR"), ("tXGB-F", "tXGB-A"),
    ("tXGB-C", "XGBoost-C"), ("tXGB-A", "XGBoost-A"), ("tXGB-F", "XGBoost-F"),  # effect of tuning
    *[(f"MLP-{s}", f"tXGB-{s}") for s in ("C", "A", "F", "D", "P0", "P1", "DR", "PA", "Z0", "G", "Z0d", "Gd")],  # R2
    ("MLP-Z0d", "MLP-Z0"), ("MLP-Gd", "MLP-G"), ("tXGB-Z0d", "tXGB-Z0"), ("tXGB-Gd", "tXGB-G"),  # R3
    ("MLP-G", "MLP-Z0"), ("tXGB-G", "tXGB-Z0"), ("MLP-G", "MLP-P0"), ("tXGB-G", "tXGB-P0"),
    *[(f"{m}-{x}", f"{m}-{y}") for m in ("tXGB", "MLP") for x, y in  # R5, Amendment 3
      (("PA", "Y1"), ("PA", "Y2"), ("PA", "K1"), ("P1", "Y2"), ("P1", "K1"), ("P0", "Y2"), ("P0", "K1"), ("Y2", "Y1"))],
    *[(f"{m}-{x}", f"{m}-{y}") for m in ("tXGB", "MLP") for x, y in  # R6, Amendment 4
      (("PA", "Y1r"), ("PA", "Y2r"), ("PA", "K1r"), ("P1", "Y2r"), ("P1", "K1r"), ("Y1r", "Y1"), ("Y2r", "Y2"), ("K1r", "K1"),
       ("PA", "PAx"), ("PA", "PAy"), ("PAx", "PAy"), ("PAy", "D"), ("PA", "D"), ("P1", "PA"), ("Y2", "C"))]]


def folds(path, prefix):
    rows = []
    for f in sorted(path.glob("*.json")):
        r = json.loads(f.read_text())
        rows.append(pd.DataFrame(dict(target=r["target"], method=prefix + r["input_set"], material_id=r["material_id"],
                                      energy_MeV=r["energy_MeV"], depth_mfp=r["depth_mfp"], label=r["label"],
                                      prediction=r["prediction"])))
    return rows


def main():
    d = [s3.load()] + folds(N / "step4/mlp_folds", "MLP-") + folds(N / "step4/xgb_folds", "tXGB-")
    for t in ("EBF", "EABF"):
        x = load_data(t)
        for name, col in (("G-P alone", "BGP_" + t), ("Reference alone", "anchor_" + t)):
            d.append(pd.DataFrame(dict(target=t, method=name, material_id=x.material_id, energy_MeV=x.energy_MeV,
                                       depth_mfp=x.depth_mfp, label=x[t], prediction=x[col])))
    d = pd.concat(d, ignore_index=True)
    n = d.groupby(["target", "method"]).material_id.nunique()
    part = n[n < 40]
    if len(part):
        print("incomplete, left out:\n", part.to_string())
    d = d.set_index(["target", "method"]).loc[n[n == 40].index].reset_index()
    m = s3.metrics(d)
    m.to_csv(N / "step4/metrics.csv", index=False)
    rng = np.random.default_rng(20261010)
    pr = pd.DataFrame([r for a, b in PAIRS for s in (None, 0.1) for r in s3.paired(d, a, b, rng, s)])
    pr["resolved"] = (pr.lo > 0) | (pr.hi < 0)
    pr.to_csv(N / "step4/paired.csv", index=False)
    d = d.assign(ae=100 * (d.prediction / d.label - 1).abs())
    by_e = d.groupby(["target", "method", "energy_MeV", "material_id"]).ae.mean().groupby(["target", "method", "energy_MeV"]).mean()
    by_e.unstack("energy_MeV").round(3).to_csv(N / "step4/mape_by_energy.csv")
    cols = ["method", "MAPE", "MAPE_0p1", "MAPE_0p15", "worst_bias_0p1", "MAPE_6_15", "rms_log", "P95", "max_abs", "worst_material"]
    L = ["# Step 4 (Amendment 2, R1-R3): nested leave-one-material-out on the 40 development compositions", "",
         "tXGB = XGBoost tuned in the inner folds; XGBoost = the article's fixed settings; sets ending in d = correction to G-P.", ""]
    for t in ("EBF", "EABF"):
        L += [f"## {t}", "", s3.md(m[m.target == t][cols].sort_values("MAPE")), ""]
    L += ["## Paired material-level MAPE differences, a minus b, percentage points, 95 % bootstrap interval over materials", "",
          s3.md(pr), ""]
    (N / "step4/RESULTS.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
