#!/usr/bin/env python3
"""Step 6 (AMENDMENT5.md sections 6-8, AMENDMENT5A.md): the single held-out evaluation.

Verifies the committed hashes, joins the frozen predictions (step6/heldout_predictions.csv) to the Geant4 labels of
the held-out cohorts and writes step6/evaluation.json, RESULTS.md and the tables. Refuses to run twice.
"""
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "validation"
N = V / "nn_tests_20261010"
S6 = N / "step6"
FR = V / "physics_surrogate_full_range_20260930"
KEYS = ["material_id", "energy_MeV", "depth_mfp"]
NB, SEED, MARGIN, GATE = 10000, 20261011, 0.25, 5.0
TREE = {"EBF": "XGBoost-F", "EABF": "tXGB-P1"}  # trees selected by the development gate


def md(t):
    t = t.copy()
    for c in t.columns:
        if t[c].dtype.kind == "f":
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    return "\n".join(["| " + " | ".join(map(str, t.columns)) + " |", "|" + "---|" * len(t.columns)] + ["| " + " | ".join(map(str, r)) + " |" for r in t.values])


def labels():
    cols = KEYS + ["EBF", "EBF_se", "EABF", "EABF_se"]
    files = [FR / "retrospective_existing/labels.csv", FR / "fresh_confirmation_amendment1/runs_low/labels.csv",
             FR / "fresh_confirmation_amendment1/runs_high/labels.csv"]
    files += sorted((V / "blind_esgyp_20261008/labels").glob("grid_s*.csv")) + sorted((V / "blind_esgyp_20261008/labels").glob("lnttl_s*.csv"))
    high = Path.home() / "BF/runs/highE_labels_20261003/labels/labels.csv"
    parts = [pd.read_csv(f)[cols] for f in files] + [pd.read_csv(high).query("material_id == 'HEH1'")[cols]]
    lab = pd.concat(parts, ignore_index=True)
    assert not lab.duplicated(KEYS).any()
    long = pd.concat([lab[KEYS + [t, t + "_se"]].rename(columns={t: "label", t + "_se": "label_se"}).assign(target=t) for t in ("EBF", "EABF")])
    return long, {str(f): hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files + [high]}


def rk(d):
    d = d.copy(); d["energy_MeV"] = d.energy_MeV.astype(float).round(6); d["depth_mfp"] = d.depth_mfp.astype(float).round(6)
    return d


def mat_err(d):
    return d.groupby("material_id").ape.mean()


def boot(x, rng):
    """x: per-material values. Mean, two-sided 95 % interval and 5th percentile over material resamples."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return dict(mean=float(x.mean()), lo=np.nan, hi=np.nan, p5=np.nan, n=len(x))
    b = x[rng.integers(0, len(x), (NB, len(x)))].mean(1)
    return dict(mean=float(x.mean()), lo=float(np.quantile(b, .025)), hi=float(np.quantile(b, .975)), p5=float(np.quantile(b, .05)), n=len(x))


def contrast(j, cohort, target, ours, other, rng, energy=None):
    d = j[(j.cohort == cohort) & (j.target == target)]
    if energy is not None:
        d = d[np.isclose(d.energy_MeV, energy)]
    a, b = mat_err(d[d.model == ours]), mat_err(d[d.model == other])
    assert list(a.index) == list(b.index)
    return dict(cohort=cohort, target=target, ours=ours, comparator=other, slice="all" if energy is None else f"{energy:g} MeV",
                ours_MAPE=float(a.mean()), comparator_MAPE=float(b.mean()), **{"improvement" if k == "mean" else k: v for k, v in boot(b - a, rng).items()})


def dev_scores():
    """Out-of-fold development scores |ln prediction - ln label| per model, target, energy, depth: 37th smallest of 40."""
    rows = []
    for kind, folder in (("MLP", "mlp_folds"), ("tXGB", "xgb_folds")):
        for s in (("DR", "PAy", "Y2", "K1r") if kind == "MLP" else ("P1", "Y2", "K1r")):
            for t in ("EBF", "EABF"):
                fs = glob.glob(str(N / f"step4/{folder}/{t}_{s}_*.json"))
                assert len(fs) == 40
                d = pd.concat([pd.DataFrame({k: json.load(open(f))[k] for k in ("energy_MeV", "depth_mfp", "label", "prediction")}) for f in fs])
                d["score"] = np.abs(np.log(d.prediction / d.label))
                q = d.groupby(["energy_MeV", "depth_mfp"]).score.apply(lambda v: np.sort(v.values)[36] if len(v) == 40 else np.nan)
                rows.append(q.reset_index().assign(target=t, model=f"{kind}-{s}"))
    return rk(pd.concat(rows, ignore_index=True))


def main():
    if (S6 / "evaluation.json").exists():
        raise SystemExit("step6/evaluation.json exists: the held-out evaluation is run once")
    man = json.loads((S6 / "frozen/manifest.json").read_text())
    bad = [f for f, h in man.items() if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h]
    assert not bad, bad
    rng = np.random.default_rng(SEED)
    pred = rk(pd.read_csv(S6 / "heldout_predictions.csv"))
    lab, lab_hash = labels()
    j = pred.merge(rk(lab), on=KEYS + ["target"], how="left", validate="many_to_one")
    ok = np.isfinite(j.label) & (j.label > 0)
    missing = j[~ok].drop_duplicates(KEYS + ["target"])[["cohort"] + KEYS + ["target"]]
    j = j[ok].copy()
    j["res"] = 100 * (j.prediction / j.label - 1); j["ape"] = j.res.abs(); j["se_pct"] = 100 * j.label_se / j.label
    j.to_csv(S6 / "heldout_joined.csv", index=False)
    out = dict(missing_label_rows=len(missing), rows_by_cohort={k: int(v) for k, v in j[(j.model == "MLP-DR") & (j.target == "EBF")].groupby("cohort").size().items()},
               labels_sha256=lab_hash, bootstrap=dict(resamples=NB, seed=SEED), margin_points=MARGIN, gate_percent=GATE)

    # primary
    prim = pd.DataFrame([contrast(j, "kedge", t, "MLP-DR", c, rng, 0.1) for t in ("EBF", "EABF") for c in ("MLP-Y2", "MLP-K1r")])
    prim["exceeds_margin"] = prim.p5 > MARGIN
    out["primary"] = dict(contrasts=prim.to_dict("records"), supported=bool(prim.exceeds_margin.all()))

    # secondary 1: gate on the K-edge cohort
    gate, depthb = [], []
    k = j[j.cohort == "kedge"]
    for (m, t, e), d in k.groupby(["model", "target", "energy_MeV"]):
        b = d.groupby("material_id").res.mean()
        w = d.loc[d.ape.idxmax()]
        gate.append(dict(model=m, target=t, energy_MeV=e, MAPE=float(mat_err(d).mean()), worst_bias=float(b.abs().max()), limiting_material=b.abs().idxmax(),
                         signed=float(b[b.abs().idxmax()]), n_materials_over_5=int((b.abs() > GATE).sum()), passes=bool(b.abs().max() <= GATE),
                         worst_point=float(w.ape), worst_point_material=w.material_id, worst_point_depth=float(w.depth_mfp), worst_point_se=float(w.se_pct),
                         rms_z=float(np.sqrt(((d.res / d.se_pct) ** 2).mean()))))
        for x, dd in d.groupby("depth_mfp"):
            depthb.append(dict(model=m, target=t, energy_MeV=e, depth_mfp=x, MAPE=float(dd.ape.mean()), mean_bias=float(dd.res.mean()),
                               worst_material_bias=float(dd.res.abs().max()), median_label_se=float(dd.se_pct.median())))
    gate, depthb = pd.DataFrame(gate), pd.DataFrame(depthb)
    gate.to_csv(S6 / "kedge_gate.csv", index=False); depthb.to_csv(S6 / "kedge_by_depth.csv", index=False)

    # secondary 2 and 3: contrasts
    sec = [contrast(j, "kedge", t, o, c, rng, e) for t in ("EBF", "EABF") for e in (0.1, 0.15)
           for o, cs in (("MLP-DR", ("MLP-Y2", "MLP-K1r", "MLP-PAy", TREE[t])), ("MLP-PAy", ("MLP-Y2", "MLP-K1r")), ("tXGB-P1", ("tXGB-Y2", "tXGB-K1r"))) for c in cs]
    sec += [contrast(j, co, t, "MLP-DR", c, rng, e) for co in ("fresh", "confirmatory") for t in ("EBF", "EABF") for e in (None, 0.1)
            for c in (TREE[t], "MLP-PAy", "MLP-Y2", "MLP-K1r")]
    sec = pd.DataFrame(sec); sec.to_csv(S6 / "contrasts.csv", index=False)

    # MAPE of every model by cohort
    rows = []
    for (co, t, m), d in j.groupby(["cohort", "target", "model"]):
        e = mat_err(d); b = boot(e, rng); d01 = d[np.isclose(d.energy_MeV, 0.1)]
        rows.append(dict(cohort=co, target=t, model=m, materials=b["n"], MAPE=b["mean"], lo=b["lo"], hi=b["hi"], MAPE_0p1=float(mat_err(d01).mean()),
                         mean_bias=float(d.res.mean()), worst_bias_0p1=float(d01.groupby("material_id").res.mean().abs().max()), max_abs=float(d.ape.max())))
    mape = pd.DataFrame(rows); mape.to_csv(S6 / "mape_by_cohort.csv", index=False)

    # secondary 5: empirical prediction intervals
    q = dev_scores()
    iv = j.merge(q, on=["energy_MeV", "depth_mfp", "target", "model"], how="inner", validate="many_to_one")
    assert iv.score.notna().all() and len(iv) == len(j[j.model.isin(q.model.unique())])
    iv["covered"] = np.abs(np.log(iv.prediction / iv.label)) <= iv.score
    iv["half_width_pct"] = 100 * (np.exp(iv.score) - 1)
    cov = iv.groupby(["cohort", "target", "model"]).agg(points=("covered", "size"), coverage=("covered", "mean"), mean_half_width_pct=("half_width_pct", "mean")).reset_index()
    cov.to_csv(S6 / "interval_coverage.csv", index=False)
    iv.groupby(["cohort", "target", "model", "energy_MeV", "depth_mfp"]).agg(points=("covered", "size"), coverage=("covered", "mean"),
                                                                           half_width_pct=("half_width_pct", "mean")).reset_index().to_csv(S6 / "interval_coverage_detail.csv", index=False)

    out["kedge_gate"] = gate.to_dict("records")
    (S6 / "evaluation.json").write_text(json.dumps(out, indent=1, allow_nan=True, default=str) + "\n")

    order = ["MLP-DR", "MLP-PAy", "MLP-Y2", "MLP-K1r", "XGBoost-F", "tXGB-P1", "tXGB-Y2", "tXGB-K1r", "Reference alone"]
    srt = lambda d: d.assign(_o=d.model.map(order.index)).sort_values(["target", "_o"]).drop(columns="_o")
    L = ["# Step 6: held-out evaluation (run once)", "",
         "Retrospective external validation after protocol freeze (AMENDMENT5.md, AMENDMENT5A.md). Percent throughout.",
         f"Label rows missing or not finite: {len(missing)}.", "",
         "## Primary test: K-edge cohort, 0.1 MeV, MLP-DR against the published-style networks", "",
         f"Improvement = comparator MAPE minus MLP-DR MAPE over 12 compositions; p5 = one-sided 95 % lower bound; margin {MARGIN} points.", "",
         md(prim[["target", "comparator", "ours_MAPE", "comparator_MAPE", "improvement", "lo", "hi", "p5", "exceeds_margin"]]), "",
         f"**Primary claim: {'SUPPORTED' if out['primary']['supported'] else 'NOT SUPPORTED'}**", "",
         "## Gate on the K-edge cohort (material-mean bias over five depths, limit 5 %)", "",
         md(srt(gate)[["target", "energy_MeV", "model", "MAPE", "worst_bias", "limiting_material", "signed", "n_materials_over_5", "passes", "worst_point", "worst_point_depth", "worst_point_se", "rms_z"]]), "",
         "## K-edge cohort by depth, 0.1 MeV, MLP-DR", "",
         md(depthb[(depthb.model == "MLP-DR") & np.isclose(depthb.energy_MeV, 0.1)].drop(columns=["model", "energy_MeV"])), "",
         "## Contrasts (improvement of 'ours' over the comparator, points of material MAPE)", "",
         md(sec[["cohort", "slice", "target", "ours", "comparator", "ours_MAPE", "comparator_MAPE", "improvement", "lo", "hi"]]), "",
         "## Every frozen model by cohort", ""]
    for co in ("kedge", "fresh", "confirmatory", "HEH1", "ESGYP", "LNTTL"):
        L += [f"### {co}", "", md(srt(mape[mape.cohort == co]).drop(columns="cohort")), ""]
    L += ["## Empirical 90 % prediction intervals (no coverage guarantee claimed)", "", md(cov), ""]
    (S6 / "RESULTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()
