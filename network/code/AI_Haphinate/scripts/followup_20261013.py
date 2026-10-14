"""Additional Geant4 calculations after the 2026-10-12 assessment of the surrogate article.

Neither run changes a registered label, model, criterion or outcome; both are descriptive.

  kedge  higher-statistics replication of the 12 K-edge compositions at 0.1 MeV with seeds
         independent of the registered 32 (seed indices from 1000): 64 seeds at every depth,
         256 at 3 mfp and 640 at 4 mfp, the tiers of the prospective test.
  ans    La and Gd, two ANS-6.4.3 elements, through the label pipeline at 0.1, 0.5, 1, 5 and
         10 MeV, for comparison with the G-P values of the standard.

  followup_20261013.py run        both runs, K-edge first; resumable
  followup_20261013.py analyze    tables in validation/followup_20261013 (only complete seeds are read)
"""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prospective_kedge as PK

V = ROOT / "validation"
OUT = V / "followup_20261013"
RUNS = PK.BF / "runs/followup_20261013"
DRIVER = PK.load_module(PK.FZ / "bf_driver.py", "followup_driver")
DEPTHS = [0.5, 1.0, 2.0, 3.0, 4.0]
ALL = ",".join(f"{d:g}" for d in DEPTHS)
EVENTS, CUT_MM, OFFSET = 10000, 0.01, 1000
KEDGE_TIERS = [(0, 64, ALL), (64, 256, "3,4"), (256, 640, "4")]
KT_TABLES = V / "response_20260928/candidate_tables"  # the tables of the registered K-edge test
ANS = {"EL_LA": ("La", 57, 6.146), "EL_GD": ("Gd", 64, 7.901)}  # element, Z, density in g/cm3
ANS_SEEDS = {0.1: 64, 0.5: 64, 1.0: 64, 5.0: 32, 10.0: 32}
PAIRS = {"P1": ("KT_GDDY", "KT_ERY2"), "P2": ("KT_EUGD", "KT_YBY4"), "P3": ("KT_PREU", "KT_YBY5"),
         "P4": ("KT_LAND", "KT_TMY6"), "C1": ("KT_EUY4", "KT_ERY6"), "C2": ("KT_SMY4", "KT_LUY7")}  # absorber, transmitter


def kedge_materials():
    rows = [r for r in csv.DictReader((ROOT / "materials/materials.csv").open()) if r["material_id"].startswith("KT_")]
    assert len(rows) == 12
    return {r["material_id"]: r for r in rows}


def ans_materials():
    return {m: dict(material_id=m, density_g_cm3=rho, **{e + "_wt_fraction": float(e == el) for e in PK.ELEMENTS})
            for m, (el, _, rho) in ANS.items()}


def ans_tables():
    import confirm_physics_full_range as CP
    folder = OUT / "response_tables"; folder.mkdir(parents=True, exist_ok=True)
    raw = V / "response_20260928/raw"
    for m, (el, _, _) in ANS.items():
        tables = {el: CP.RT.parse_table((raw / f"z{CP.RT.Z[el]:02d}.html").read_text())}
        PK.response_table(CP.RT, tables, {el: 1.0}, folder / f"{m}.csv")
    return folder


def plan():
    """(set, material, energy text, seed index, depths text)"""
    rows = [("kedge", m, "0.1", OFFSET + i, d) for a, b, d in KEDGE_TIERS for m in kedge_materials() for i in range(a, b)]
    rows += [("ans", m, f"{e:g}", OFFSET + i, ALL) for e, n in ANS_SEEDS.items() for m in ANS for i in range(n)]
    return rows


def run(a):
    RUNS.mkdir(parents=True, exist_ok=True)
    binary = RUNS / "BF_snapshot"
    if not binary.exists():
        shutil.copy2(PK.LABEL_BINARY, binary)
    assert PK.digest(binary) == PK.LABEL_BINARY_SHA
    os.environ["G4LEDATA"] = str(PK.G4DATA)
    base = {k: v for k, v in os.environ.items() if not k.startswith(("BF_", "NIST_"))}
    base.update(BF_FIXED_EVENTS="1", BF_VR_MODE="off", BF_RESPONSE_REQUIRE_TABLE="1", BF_RESPONSE_BOUND="lower")
    env = {"kedge": dict(base, BF_RESPONSE_TABLE_DIR=str(KT_TABLES.resolve())),
           "ans": dict(base, BF_RESPONSE_TABLE_DIR=str(ans_tables().resolve()))}
    mats = {"kedge": kedge_materials(), "ans": ans_materials()}
    # the macro of this driver reproduces the registered one: same text for a registered seed index
    old = PK.BF / "runs/kedge_test_20260928/fluo/KT_LAND/E0.1/seed_000/m.txt"
    assert DRIVER.material_macro(mats["kedge"]["KT_LAND"], "0.1", ALL.split(","), EVENTS, DRIVER.seed_pair("KT_LAND", "0.1", 0), CUT_MM, None) == old.read_text()

    def one(r):
        s, m, e, i, depths = r
        args = SimpleNamespace(output=RUNS / s, mfps=depths.split(","), events=EVENTS, cut_mm=CUT_MM, shell_fraction=None)
        try:
            DRIVER.run_seed(args, binary, env[s], mats[s][m], e, i)
        except Exception as err:
            return f"FAILED {s} {m} E{e} seed {i}: {err}"

    for s in ("kedge", "ans"):
        rows = [r for r in plan() if r[0] == s]
        rows.sort(key=lambda r: (r[4].count(","), float(r[2]), r[1], r[3]))
        print(f"{s}: {len(rows)} seed processes, started {datetime.now(timezone.utc).isoformat()}", flush=True)
        for attempt in range(3):
            t0 = time.time(); failed = []
            with ThreadPoolExecutor(max_workers=a.jobs) as pool:
                for k, msg in enumerate(pool.map(one, rows), 1):
                    if msg:
                        failed.append(msg); print(msg, flush=True)
                    if k % 500 == 0:
                        print(f"{s}: {k}/{len(rows)} seed processes, {time.time() - t0:.0f} s", flush=True)
            if not failed:
                break
        if failed:
            raise SystemExit(f"{s}: seed processes still failing")
        print(f"{s.upper()}_DONE {datetime.now(timezone.utc).isoformat()}", flush=True)
    print("ALL_DONE", flush=True)


def read(s):
    """Per-seed values of every complete planned seed process of one set."""
    out = []
    for _, m, e, i, depths in (r for r in plan() if r[0] == s):
        marker = RUNS / s / m / f"E{e}" / f"seed_{i:03d}" / "complete.json"
        if not marker.exists():
            continue
        for x in json.loads(marker.read_text())["rows"]:
            out.append(dict(material_id=m, energy_MeV=float(e), depth_mfp=float(x["MFP"]), index=i, EBF=float(x["EBF_TL"]),
                            EABF=float(x["EABF_TL"]), transport_ratio=float(x["TransportRatio"])))
    return pd.DataFrame(out)


def points(seeds):
    rows = []
    for key, g in seeds.groupby(PK.KEYS):
        n = len(g); row = dict(zip(PK.KEYS, key), seeds=n, transport_ratio=g.transport_ratio.mean())
        for t in ("EBF", "EABF"):
            v = g[t].values
            row.update({t: v.mean(), t + "_se": v.std(ddof=1) / math.sqrt(n), t + "_max_seed_share": v.max() / v.sum()})
        rows.append(row)
    return pd.DataFrame(rows)


def gp(p, x):
    k = p["c"] * x ** p["a"] + p["d"] * (math.tanh(x / p["xk"] - 2) - math.tanh(-2)) / (1 - math.tanh(-2))
    return 1 + (p["b"] - 1) * x if abs(k - 1) < 1e-9 else 1 + (p["b"] - 1) * (k ** x - 1) / (k - 1)


def analyze(a):
    OUT.mkdir(exist_ok=True)
    text = ["# Additional calculations, 2026-10-13", "", "Descriptive; no registered label, model or outcome is changed.", ""]
    seeds = read("kedge")
    if len(seeds):
        new = points(seeds)
        old = pd.read_csv(V / "kedge_test_20260928/result_per_seed.csv")
        old = old[old.energy_MeV == 0.1].rename(columns={"MFP": "depth_mfp", "EBF_TL": "EBF", "EABF_TL": "EABF"})
        old["transport_ratio"] = np.nan
        old = points(old.assign(index=old.seed))
        j = new.merge(old, on=PK.KEYS, suffixes=("", "_reg"))
        for t in ("EBF", "EABF"):
            j[t + "_reg_minus_new_pct"] = 100 * (j[t + "_reg"] / j[t] - 1)
            j[t + "_reg_se"] = j[t + "_se_reg"]
            j[t + "_z"] = (j[t + "_reg"] - j[t]) / np.hypot(j[t + "_reg_se"], j[t + "_se"])
        j.to_csv(OUT / "kedge_labels.csv", index=False)
        text += ["## K-edge compositions at 0.1 MeV: registered 32 seeds against the independent replication", "",
                 f"Seeds read: {len(seeds)} values. Replication seeds at 4 mfp: {int(new[new.depth_mfp == 4].seeds.min())} to {int(new[new.depth_mfp == 4].seeds.max())}.", ""]
        d4 = j[j.depth_mfp == 4][["material_id", "seeds", "EBF", "EBF_se", "EBF_reg", "EBF_reg_se", "EBF_z", "EABF", "EABF_se", "EABF_reg", "EABF_reg_se", "EABF_z"]]
        text += ["### Labels at 4 mfp", "", PK.md(d4), ""]
        zz = np.concatenate([j.EBF_z.values, j.EABF_z.values])
        text += [f"All 120 label comparisons: |z| > 2 in {int((abs(zz) > 2).sum())}, |z| > 3 in {int((abs(zz) > 3).sum())}; largest |z| {abs(zz).max():.2f}.", ""]
        rows = []
        for pair, (ab, tr) in PAIRS.items():
            for d in DEPTHS:
                row = dict(pair=pair, depth_mfp=d)
                for tag, tab, suf in (("new", new, ""), ("registered", old, "")):
                    A = tab[(tab.material_id == ab) & (tab.depth_mfp == d)].iloc[0]; T = tab[(tab.material_id == tr) & (tab.depth_mfp == d)].iloc[0]
                    for t in ("EBF", "EABF"):
                        delta = math.log(T[t] / A[t]); se = math.hypot(T[t + "_se"] / T[t], A[t + "_se"] / A[t])
                        row.update({f"{t}_{tag}_pct": 100 * (math.exp(delta) - 1), f"{t}_{tag}_se_pct": 100 * math.exp(delta) * se, f"{t}_{tag}_z": delta / se})
                rows.append(row)
        pc = pd.DataFrame(rows); pc.to_csv(OUT / "kedge_pair_contrasts.csv", index=False)
        text += ["### Transmitter excess over the absorber, percent", "", PK.md(pc[pc.depth_mfp.isin([0.5, 4])]), ""]
        pred = pd.read_csv(V / "nn_tests_20261010/step6/heldout_joined.csv")
        pred = pred[(pred.cohort == "kedge") & np.isclose(pred.energy_MeV, 0.1)].merge(
            new.rename(columns={"EBF": "new_EBF", "EABF": "new_EABF", "EBF_se": "new_EBF_se", "EABF_se": "new_EABF_se"}), on=PK.KEYS)
        pred["new"] = np.where(pred.target == "EBF", pred.new_EBF, pred.new_EABF)
        pred["new_se_pct"] = 100 * np.where(pred.target == "EBF", pred.new_EBF_se, pred.new_EABF_se) / pred["new"]
        pred["ape_new"] = 100 * abs(pred.prediction / pred["new"] - 1); pred["res_new"] = 100 * (pred.prediction / pred["new"] - 1)
        score = pred.groupby(["target", "model"]).agg(MAPE_registered=("ape", "mean"), MAPE_replication=("ape_new", "mean"), bias_replication=("res_new", "mean")).reset_index()
        worst = pred.groupby(["target", "model", "material_id"]).res_new.mean().abs().groupby(["target", "model"]).max().rename("worst_material_bias_replication").reset_index()
        score = score.merge(worst); score.to_csv(OUT / "kedge_model_scores.csv", index=False)
        text += ["### Fixed predictions of the network study scored against both label sets, 0.1 MeV, 12 compositions, percent", "", PK.md(score), ""]
        dr = pred[pred.model == "MLP-DR"].groupby(["target", "depth_mfp"]).agg(MAPE_registered=("ape", "mean"), MAPE_replication=("ape_new", "mean"),
              bias_replication=("res_new", "mean"), median_label_se_registered=("se_pct", "median"), median_label_se_replication=("new_se_pct", "median")).reset_index()
        dr.to_csv(OUT / "kedge_fourinput_by_depth.csv", index=False)
        text += ["### Four-input network by depth", "", PK.md(dr), ""]
    seeds = read("ans")
    if len(seeds):
        sys.path.insert(0, str(ROOT.parent / "ai_shielding/python/analysis"))
        import gp_coefficients as G
        p = points(seeds)
        for t, kind in (("EBF", "exposure"), ("EABF", "absorption")):
            p[t + "_GP"] = [gp(G.GP_COEFFICIENTS[kind][ANS[m][1]][e], x) for m, e, x in zip(p.material_id, p.energy_MeV, p.depth_mfp)]
            p[t + "_dev_pct"] = 100 * (p[t] / p[t + "_GP"] - 1)
        p.to_csv(OUT / "ans_benchmark.csv", index=False)
        text += ["## La and Gd against the ANS-6.4.3 G-P values", "",
                 PK.md(p[["material_id", "energy_MeV", "depth_mfp", "seeds", "transport_ratio", "EBF", "EBF_se", "EBF_GP", "EBF_dev_pct", "EABF", "EABF_se", "EABF_GP", "EABF_dev_pct"]]), ""]
    (OUT / "RESULTS_DRAFT.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "analyze", "plan"])
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    if a.action == "plan":
        from collections import Counter
        print(Counter((r[0], r[2], r[4]) for r in plan()))
    else:
        {"run": run, "analyze": analyze}[a.action](a)
