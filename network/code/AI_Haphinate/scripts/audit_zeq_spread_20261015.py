#!/usr/bin/env python3
"""Residual spread beyond a quadratic in Zeq at 0.1 MeV, by depth (audit, 2026-10-15).

Reconciles the depth-averaged spread quoted in the article (4.6 % EBF, 3.3 % EABF over 41 compositions) with the
matched-pair contrasts: per depth, the standard deviation of the residuals in ln B, the RMS relative label error, and
the range between the largest and smallest residual. Writes validation/audit_20261015/zeq_residual_by_depth.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "validation"
tr = pd.read_csv(V / "highE_extension_20261003/surrogate_17E/training_data.csv")
d = tr[np.isclose(tr.energy_MeV, 0.1)][["material_id", "depth_mfp", "EBF", "EABF", "EBF_se", "EABF_se", "Zeq"]]
h = pd.read_csv(V / "nn_tests_20261010/step6/heldout_joined.csv")
h = h[(h.material_id == "HEH1") & (h.model == "MLP-DR") & np.isclose(h.energy_MeV, 0.1)]
h = h.pivot_table(index="depth_mfp", columns="target", values=["label", "label_se"]).reset_index()
g = pd.read_csv(V / "gp_baseline_20261014/gp_values.csv")
zh = g[(g.material_id == "HEH1") & np.isclose(g.energy_MeV, 0.1)].Zeq.iloc[0]
d = pd.concat([d, pd.DataFrame({"material_id": "HEH1", "depth_mfp": h.depth_mfp, "EBF": h[("label", "EBF")], "EABF": h[("label", "EABF")],
                                "EBF_se": h[("label_se", "EBF")], "EABF_se": h[("label_se", "EABF")], "Zeq": zh})])
rows = []
for t in ("EBF", "EABF"):
    for x, gg in d.groupby("depth_mfp"):
        y = np.log(gg[t]); r = y - np.polyval(np.polyfit(gg.Zeq, y, 2), gg.Zeq)
        rows.append(dict(target=t, depth_mfp=x, compositions=len(gg), residual_sd_pct=100 * r.std(ddof=3),
                         label_rms_se_pct=100 * np.sqrt(np.mean((gg[t + "_se"] / gg[t]) ** 2)),
                         extreme_range_pct=100 * (np.exp(r.max() - r.min()) - 1)))
o = pd.DataFrame(rows)
o.to_csv(V / "audit_20261015/zeq_residual_by_depth.csv", index=False)
print(o.round(2).to_string(index=False)); print(o.groupby("target").residual_sd_pct.mean().round(2))
