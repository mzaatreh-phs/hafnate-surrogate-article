"""Sensitivity of the 0.1 MeV labels to the fluorescence line energies, within 0.5 to 4 mfp.

Descriptive; no registered label, model, criterion or outcome is changed.

Geant4 takes its fluorescence line energies from the EADL files of G4EMLOW (fluor). In those
files the Hf K-M3 line, 63.383 keV, lies above the Lu K edge of the transport, 63.320 keV; in
the Bearden files (fluor_Bearden) it lies below, 63.234 keV. The labels are recalculated with
the Bearden files (BF_FLUO_DIR=bearden), all other settings those of the labels:

  LUH, TYL          development compositions with Lu; both data sets are run here
  KT_LUY7           K-edge transmitter with Lu; Bearden here, EADL from the replication of 2026-10-13
  KT_TMY6, KT_LAND  controls without Lu, a transmitter and an absorber; Bearden here, EADL as above

Seed indices start at 2000, independent of every earlier run: 64 seeds at the five depths,
256 at 3 mfp and 640 at 4 mfp.

  bearden_check_20261013.py run        resumable
  bearden_check_20261013.py analyze    tables in validation/bearden_check_20261013 (only complete seeds are read)
"""
import argparse
import csv
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
OUT = V / "bearden_check_20261013"
RUNS = PK.BF / "runs/bearden_check_20261013"
REPLICATION = PK.BF / "runs/followup_20261013/kedge"
DRIVER = PK.load_module(PK.FZ / "bf_driver.py", "bearden_driver")
DEPTHS = [0.5, 1.0, 2.0, 3.0, 4.0]
ALL = ",".join(f"{d:g}" for d in DEPTHS)
EVENTS, CUT_MM, OFFSET = 10000, 0.01, 2000
TIERS = [(0, 64, ALL), (64, 256, "3,4"), (256, 640, "4")]
TABLES = V / "response_20260928/candidate_tables"
SETS = {"bearden": ["LUH", "TYL", "KT_LUY7", "KT_TMY6", "KT_LAND"], "eadl": ["LUH", "TYL"]}
LU = {"LUH": True, "TYL": True, "KT_LUY7": True, "KT_TMY6": False, "KT_LAND": False}


def materials():
    rows = {r["material_id"]: r for r in csv.DictReader((ROOT / "materials/materials.csv").open())}
    return {m: rows[m] for m in LU}


def plan():
    """(data set, material, seed index, depths text)"""
    return [(s, m, OFFSET + i, d) for s, ms in SETS.items() for a, b, d in TIERS for m in ms for i in range(a, b)]


def run(a):
    RUNS.mkdir(parents=True, exist_ok=True)
    binary = RUNS / "BF_snapshot"
    if not binary.exists():
        shutil.copy2(PK.LABEL_BINARY, binary)
    assert PK.digest(binary) == PK.LABEL_BINARY_SHA
    os.environ["G4LEDATA"] = str(PK.G4DATA)
    base = {k: v for k, v in os.environ.items() if not k.startswith(("BF_", "NIST_"))}
    base.update(BF_FIXED_EVENTS="1", BF_VR_MODE="off", BF_RESPONSE_REQUIRE_TABLE="1", BF_RESPONSE_BOUND="lower",
                BF_RESPONSE_TABLE_DIR=str(TABLES.resolve()))
    env = {s: dict(base, BF_FLUO_DIR=s) for s in SETS}
    mats = materials()

    def one(r):
        s, m, i, depths = r
        args = SimpleNamespace(output=RUNS / s, mfps=depths.split(","), events=EVENTS, cut_mm=CUT_MM, shell_fraction=None)
        try:
            DRIVER.run_seed(args, binary, env[s], mats[m], "0.1", i)
        except Exception as err:
            return f"FAILED {s} {m} seed {i}: {err}"

    rows = sorted(plan(), key=lambda r: (r[3].count(","), r[0], r[1], r[2]))
    rows = rows[:a.limit] if a.limit else rows
    print(f"{len(rows)} seed processes, started {datetime.now(timezone.utc).isoformat()}", flush=True)
    for attempt in range(3):
        t0 = time.time(); failed = []
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for k, msg in enumerate(pool.map(one, rows), 1):
                if msg:
                    failed.append(msg); print(msg, flush=True)
                if k % 500 == 0:
                    print(f"{k}/{len(rows)} seed processes, {time.time() - t0:.0f} s", flush=True)
        if not failed:
            break
    if failed:
        raise SystemExit("seed processes still failing")
    if not a.limit:
        print("ALL_DONE", flush=True)


def read(folder, material, first):
    """Per-seed values of every complete seed process of one material, seed indices from `first`."""
    out = []
    for marker in sorted((folder / material / "E0.1").glob("seed_*/complete.json")):
        i = int(marker.parent.name.split("_")[1])
        if i < first:
            continue
        for x in json.loads(marker.read_text())["rows"]:
            out.append(dict(material_id=material, depth_mfp=float(x["MFP"]), index=i, EBF=float(x["EBF_TL"]), EABF=float(x["EABF_TL"])))
    return pd.DataFrame(out)


def points(seeds):
    rows = []
    for (m, d), g in seeds.groupby(["material_id", "depth_mfp"]):
        row = dict(material_id=m, depth_mfp=d, seeds=len(g))
        for t in ("EBF", "EABF"):
            v = g[t].values
            row.update({t: v.mean(), t + "_se": v.std(ddof=1) / math.sqrt(len(v))})
        rows.append(row)
    return pd.DataFrame(rows)


def data_sets_used():
    """The fluorescence data set each seed log reports, as a check of the switch."""
    seen = {}
    for s in SETS:
        for log in list((RUNS / s).glob("*/E0.1/seed_*/*.log"))[:200]:
            for line in log.read_text(errors="replace").splitlines():
                if line.startswith("BF: fluorescence data set"):
                    seen.setdefault(s, set()).add(line.split()[-1])
    return seen


def analyze(a):
    OUT.mkdir(exist_ok=True)
    text = ["# Bearden line energies against the EADL default, 0.1 MeV, 2026-10-13", "",
            "Descriptive; no registered label, model or outcome is changed.", "",
            f"Fluorescence data set reported by the seed logs: {data_sets_used()}", ""]
    b = pd.concat([read(RUNS / "bearden", m, OFFSET) for m in SETS["bearden"]])
    e = pd.concat([read(RUNS / "eadl", m, OFFSET) for m in SETS["eadl"]] +
                  [read(REPLICATION, m, 1000) for m in SETS["bearden"] if m not in SETS["eadl"]])
    if not len(b) or not len(e):
        print("no complete seeds yet"); return
    j = points(b).merge(points(e), on=["material_id", "depth_mfp"], suffixes=("_bearden", "_eadl"))
    j.insert(1, "contains_Lu", j.material_id.map(LU))
    for t in ("EBF", "EABF"):
        r = j[t + "_bearden"] / j[t + "_eadl"]
        se = r * np.hypot(j[t + "_se_bearden"] / j[t + "_bearden"], j[t + "_se_eadl"] / j[t + "_eadl"])
        j[t + "_change_pct"] = 100 * (r - 1); j[t + "_change_se_pct"] = 100 * se; j[t + "_z"] = (r - 1) / se
    j.to_csv(OUT / "bearden_vs_eadl.csv", index=False)
    cols = ["material_id", "contains_Lu", "depth_mfp", "seeds_bearden", "seeds_eadl"]
    for t in ("EBF", "EABF"):
        text += [f"## {t}: Bearden against EADL", "",
                 PK.md(j[cols + [t + "_eadl", t + "_se_eadl", t + "_bearden", t + "_se_bearden", t + "_change_pct", t + "_change_se_pct", t + "_z"]]), ""]
    # saved four-input predictions for the three K-edge compositions, scored against both label sets
    pred = pd.read_csv(V / "nn_tests_20261010/step6/heldout_joined.csv")
    pred = pred[(pred.cohort == "kedge") & np.isclose(pred.energy_MeV, 0.1) & (pred.model == "MLP-DR")].merge(j, on=["material_id", "depth_mfp"])
    if len(pred):
        for s in ("eadl", "bearden"):
            lab = np.where(pred.target == "EBF", pred["EBF_" + s], pred["EABF_" + s])
            pred["res_" + s] = 100 * (pred.prediction / lab - 1)
        sc = pred.groupby(["material_id", "target"]).agg(MAPE_eadl=("res_eadl", lambda x: abs(x).mean()), bias_eadl=("res_eadl", "mean"),
                                                         MAPE_bearden=("res_bearden", lambda x: abs(x).mean()), bias_bearden=("res_bearden", "mean")).reset_index()
        sc.to_csv(OUT / "fourinput_scores.csv", index=False)
        text += ["## Saved four-input predictions against both label sets, percent", "", PK.md(sc), ""]
    (OUT / "RESULTS_DRAFT.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["run", "analyze", "plan"])
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N seed processes (smoke test)")
    a = ap.parse_args()
    if a.action == "plan":
        from collections import Counter
        print(Counter((r[0], r[1], r[3]) for r in plan()), len(plan()))
    else:
        {"run": run, "analyze": analyze}[a.action](a)
