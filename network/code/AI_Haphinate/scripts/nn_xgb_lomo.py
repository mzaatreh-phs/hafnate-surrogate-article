#!/usr/bin/env python3
"""Amendment 2, R2 and R3: XGBoost tuned inside the outer leave-one-material-out folds.

Same data, input sets, target weights, outer folds and inner grouped 5-fold split as nn_mlp_lomo.py. Search fixed in
AMENDMENT2.md: max_depth {3, 5} x learning_rate {0.03, 0.1} x min_child_weight {1, 5}; at most 1,500 trees, early
stopping (patience 50) on the weighted inner validation loss; final number of trees = mean best iteration.

Usage: nn_xgb_lomo.py [--jobs 14] [--sets C A ...]
"""
import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nn_mlp_lomo import DATA, GP, SETS, load, weights  # noqa: E402

OUT = GP.parent / "xgb_folds"
GRID = list(itertools.product([3, 5], [0.03, 0.1], [1, 5]))  # max_depth, learning_rate, min_child_weight
ALL = ["C", "A", "F", "D", "P0", "P1", "DR", "PA", "Z0", "G", "Z0d", "Gd"]


def wts(d):
    """Protocol weights rescaled to mean 1, so that min_child_weight counts rows as usual."""
    w = weights(d)
    return w * len(w) / w.sum()


def model(cfg, n, stop=None):
    return XGBRegressor(n_estimators=n, max_depth=cfg[0], learning_rate=cfg[1], min_child_weight=cfg[2], subsample=0.9,
                        colsample_bytree=0.9, random_state=42, objective="reg:squarederror", n_jobs=1,
                        early_stopping_rounds=stop)


def task(args):
    target, sname, mat = args
    out = OUT / f"{target}_{sname}_{mat}.json"
    if out.exists():
        return str(out)
    t0 = time.time()
    d = load(target, sname)
    cols = SETS[sname]
    tr, te = d[d.material_id != mat].reset_index(drop=True), d[d.material_id == mat].reset_index(drop=True)
    mats = sorted(tr.material_id.unique())
    perm = np.random.default_rng(20261010).permutation(len(mats))  # same inner split as the networks
    inner = [set(np.array(mats)[perm[k::5]]) for k in range(5)]
    table = []
    for cfg in GRID:
        losses, its = [], []
        for val in inner:
            a, b = tr[~tr.material_id.isin(val)], tr[tr.material_id.isin(val)]
            wb = wts(b)
            m = model(cfg, 1500, 50).fit(a[cols], a.y, sample_weight=wts(a), eval_set=[(b[cols], b.y)],
                                         sample_weight_eval_set=[wb], verbose=False)
            p = m.predict(b[cols])
            losses.append(float((wb * (p - b.y.values) ** 2).sum() / wb.sum())); its.append(m.best_iteration + 1)
        table.append(dict(max_depth=cfg[0], learning_rate=cfg[1], min_child_weight=cfg[2],
                          inner_loss=float(np.mean(losses)), trees=int(round(np.mean(its)))))
    best = min(table, key=lambda r: r["inner_loss"])
    m = model((best["max_depth"], best["learning_rate"], best["min_child_weight"]), max(20, best["trees"]))
    m.fit(tr[cols], tr.y, sample_weight=wts(tr))
    pred = np.exp(m.predict(te[cols]) + te.offset.values)
    res = dict(target=target, input_set=sname, material_id=mat, chosen=best, grid=table, seconds=time.time() - t0,
               energy_MeV=te.energy_MeV.tolist(), depth_mfp=te.depth_mfp.tolist(), label=te[target].tolist(),
               label_se=te[target + "_se"].tolist(), prediction=pred.tolist())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(res)); os.replace(tmp, out)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--sets", nargs="+", default=ALL)
    a = ap.parse_args()
    mats = sorted(pd.read_csv(DATA).material_id.unique())
    tasks = [(t, s, m) for t in ("EBF", "EABF") for s in a.sets for m in mats]
    t0 = time.time()
    with Pool(a.jobs) as pool:
        for i, f in enumerate(pool.imap_unordered(task, tasks), 1):
            print(f"{i}/{len(tasks)} {Path(f).name} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
