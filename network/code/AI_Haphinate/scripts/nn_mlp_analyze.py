#!/usr/bin/env python3
"""Metrics for step 3 of validation/nn_tests_20261010 (PROTOCOL.md section 5, Amendment 1).

Material-balanced MAPE, 0.1 and 0.15 MeV slices, worst material bias at 0.1 MeV, RMS log error, tails, 6-15 MeV, and
paired material-level differences against the tree and neighbour baselines of the identical folds
(validation/highE_extension_20261003/surrogate_17E/holdout_predictions.csv), with bootstrap intervals over materials.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "validation/nn_tests_20261010/step3_mlp"
BASE = {"XGB": "XGBoost-C", "3NN": "Three neighbours", "atomic_features": "XGBoost-A", "full_features": "XGBoost-F"}


def load():
    rows = []
    for f in sorted((D / "folds").glob("*.json")):
        r = json.loads(f.read_text())
        rows.append(pd.DataFrame(dict(target=r["target"], method="MLP-" + r["input_set"], material_id=r["material_id"],
                                      energy_MeV=r["energy_MeV"], depth_mfp=r["depth_mfp"], label=r["label"],
                                      prediction=r["prediction"])))
    nn = pd.concat(rows, ignore_index=True)
    h = pd.read_csv(ROOT / "validation/highE_extension_20261003/surrogate_17E/holdout_predictions.csv")
    h = h[h.method.isin(BASE)].assign(method=lambda d: d.method.map(BASE))[nn.columns]
    return pd.concat([nn, h], ignore_index=True)


def metrics(d):
    d = d.assign(e=d.prediction / d.label - 1)
    d["ae"] = 100 * d.e.abs()
    out = []
    for (t, m), g in d.groupby(["target", "method"]):
        mat = g.groupby("material_id").ae.mean()
        lo, l15, hi = g[np.isclose(g.energy_MeV, 0.1)], g[np.isclose(g.energy_MeV, 0.15)], g[g.energy_MeV >= 6]
        out.append(dict(target=t, method=m, materials=g.material_id.nunique(), MAPE=mat.mean(),
                        MAPE_0p1=lo.groupby("material_id").ae.mean().mean(), MAPE_0p15=l15.groupby("material_id").ae.mean().mean(),
                        worst_bias_0p1=(100 * lo.groupby("material_id").e.mean()).abs().max(),
                        MAPE_6_15=hi.groupby("material_id").ae.mean().mean(),
                        rms_log=np.sqrt(np.mean(np.log(g.prediction / g.label) ** 2)) * 100,
                        P95=g.ae.quantile(0.95), max_abs=g.ae.max(), worst_material=mat.max()))
    return pd.DataFrame(out)


def paired(d, a, b, rng, slice_=None):
    """Material-level MAPE difference a - b with a 95 % bootstrap interval over materials."""
    d = d.assign(ae=100 * (d.prediction / d.label - 1).abs())
    if slice_ is not None:
        d = d[np.isclose(d.energy_MeV, slice_)]
    out = []
    for t in ("EBF", "EABF"):
        p = d[d.target == t].groupby(["method", "material_id"]).ae.mean().unstack(0)
        if a not in p or b not in p:
            continue
        diff = (p[a] - p[b]).dropna().values
        boot = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(1)
        out.append(dict(target=t, a=a, b=b, slice="all" if slice_ is None else f"{slice_} MeV", n=len(diff), diff=diff.mean(),
                        lo=np.quantile(boot, 0.025), hi=np.quantile(boot, 0.975)))
    return out


def md(t):
    """Markdown table without the optional tabulate dependency."""
    rows = [list(map(str, t.columns)), ["---"] * len(t.columns)] + [[f"{v:.3f}" if isinstance(v, float) else str(v) for v in r] for r in t.itertuples(index=False)]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def main():
    d = load()
    n = d[d.method.str.startswith("MLP")].groupby(["target", "method"]).material_id.nunique()
    print("folds complete per network:\n", n.to_string())
    m = metrics(d)
    m.to_csv(D / "metrics.csv", index=False)
    rng = np.random.default_rng(20261010)
    pr = []
    for a, b in [("MLP-P1", "MLP-F"), ("MLP-P1", "MLP-C"), ("MLP-P1", "XGBoost-F"), ("MLP-P1", "XGBoost-C"), ("MLP-P1", "Three neighbours"),
                 ("MLP-P0", "XGBoost-F"), ("MLP-D", "XGBoost-C"), ("MLP-F", "XGBoost-F"), ("MLP-C", "XGBoost-C")]:
        for s in (None, 0.1):
            pr += paired(d, a, b, rng, s)
    pr = pd.DataFrame(pr); pr["resolved"] = (pr.lo > 0) | (pr.hi < 0)
    pr.to_csv(D / "paired.csv", index=False)
    cols = ["method", "MAPE", "MAPE_0p1", "MAPE_0p15", "worst_bias_0p1", "MAPE_6_15", "rms_log", "P95", "max_abs", "worst_material"]
    L = ["# Step 3: small MLP, nested leave-one-material-out on the 40 development compositions", ""]
    for t in ("EBF", "EABF"):
        L += [f"## {t}", "", md(m[m.target == t][cols].sort_values("MAPE")), ""]
    L += ["## Paired material-level MAPE differences, a minus b, percentage points, 95 % bootstrap interval over materials", "",
          md(pr), ""]
    (D / "RESULTS.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
