#!/usr/bin/env python3
"""Step 3 of validation/nn_tests_20261010/PROTOCOL.md: small MLP, direct ln B target, nested leave-one-material-out.

Development shallow labels only (40 compositions x 17 energies x 8 depths). For each outer fold the architecture is
chosen on an inner grouped 5-fold split of the 39 training materials (grid fixed in the protocol), then an ensemble
of 10 seeds is trained on the 39 materials and predicts the omitted one. Final training length: the mean best epoch
of the chosen configuration over the inner folds (the stopping rule needs no label of the omitted material).
Amendment 1 (AMENDMENT1.md): mini-batch training and three input sets without mass fractions.
One task = (target, input set, outer material); results are written per task so the run can resume.

Usage: nn_mlp_lomo.py [--jobs 16] [--sets DR PA ...] [--only TARGET SET MATERIAL]
"""
import argparse
import itertools
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/nn_tests_20261010/step3_mlp"
DATA = ROOT / "validation/highE_extension_20261003/surrogate_17E/training_data.csv"
WT = ["La_wt", "Ce_wt", "Pr_wt", "Nd_wt", "Sm_wt", "Eu_wt", "Gd_wt", "Tb_wt", "Dy_wt", "Ho_wt", "Er_wt", "Tm_wt", "Y_wt",
      "Yb_wt", "Lu_wt", "Sc_wt", "Hf_wt", "O_wt"]
ATOMIC = ["log_fluorescence_number_yield", "fluorescence_energy_yield", "log_yield_weighted_attenuation",
          "log_yield_harmonic_attenuation", "line_energy_mean_fraction", "line_below_edge_fraction",
          "line_edge_gap_mean_keV", "line_edge_gap_abs_mean_keV", "Hf_Kalpha_attenuation_ratio", "Hf_Kbeta_attenuation_ratio"]
C = ["log10_E_MeV", "depth_mfp", "density_g_cm3"] + WT
BASE = ["log10_E_MeV", "depth_mfp"]
SETS = {"C": C, "A": C + ATOMIC, "F": C + ATOMIC + ["log_anchor"],  # registered
        "D": BASE + ["density_g_cm3"], "P0": BASE + ["log_anchor"],  # Amendment 1: no mass fractions
        "P1": BASE + ["density_g_cm3"] + ATOMIC + ["log_anchor"],
        "DR": BASE + ["density_g_cm3", "log_anchor"], "PA": BASE + ["density_g_cm3"] + ATOMIC,  # Amendment 2, R1
        "Z0": BASE + ["Zeq_epics", "log_bgp"], "G": BASE + ["Zeq_epics", "log_bgp", "log_anchor"]}  # Amendment 2, R3
SETS.update({k + "d": SETS[k] for k in ("Z0", "G")})  # same inputs, target ln(B / B_GP)
SETS.update({  # Amendment 3, R5: published-style input sets (comparators, not reproductions)
    "Y1": BASE + ["Zeq_epics", "log_mu", "EK_zeq", "mek_zeq"],
    "Y2": BASE + ["Zeq_epics", "log_mu", "EK_mean", "mek_mean", "EK_min", "mek_min", "EK_max", "mek_max"],
    "K1": BASE + ["log_S_inc", "log_S_pe", "log_S_tot"]})
LINE_EDGE = ["line_energy_mean_fraction", "line_below_edge_fraction", "line_edge_gap_mean_keV", "line_edge_gap_abs_mean_keV"]
HF_RATIO = ["Hf_Kalpha_attenuation_ratio", "Hf_Kbeta_attenuation_ratio"]
SETS.update({k + "r": SETS[k] + ["density_g_cm3"] for k in ("Y1", "Y2", "K1")})  # Amendment 4, R6: density-matched
SETS["PAx"] = [c for c in SETS["PA"] if c not in LINE_EDGE]  # Amendment 4, R6: descriptor ablation
SETS["PAy"] = [c for c in SETS["PAx"] if c not in HF_RATIO]
LIT = ROOT / "validation/nn_tests_20261010/step4/lit_features.csv"
STEP3 = ["C", "A", "F", "D", "P0", "P1"]
GP = ROOT / "validation/nn_tests_20261010/step4/gp_features.csv"
GRID = list(itertools.product([32, 64, 128], [2, 3], [1e-5, 1e-4, 1e-3]))  # width, depth, weight decay
MAX_EPOCHS, PATIENCE, LR, S0, N_ENS, BATCH = 300, 30, 1e-3, 0.005, 10, 128  # Amendment 1


def load(target, sname="C"):
    d = pd.read_csv(DATA)
    d["log_anchor"] = np.log(d["anchor_" + target])
    d = d.merge(pd.read_csv(GP), on=["material_id", "energy_MeV", "depth_mfp"], how="left", validate="one_to_one")
    if LIT.exists():
        d = d.merge(pd.read_csv(LIT), on=["material_id", "energy_MeV", "depth_mfp"], how="left", validate="one_to_one")
        assert len(d) == 5440 and d.log_mu.notna().all()
    d["log_bgp"] = np.log(d["BGP_" + target])
    d["offset"] = d.log_bgp if sname.endswith("d") else 0.0  # correction target for the "d" sets
    d["y"] = np.log(d[target]) - d.offset
    d["rel"] = d[target + "_se"] / d[target]
    return d


def weights(d):
    """Known-noise weights, capped at the 95th percentile, each material with equal total weight."""
    w = 1.0 / (d.rel.values ** 2 + S0 ** 2)
    w = np.minimum(w, np.quantile(w, 0.95))
    tot = pd.Series(w).groupby(d.material_id.values).transform("sum").values
    return w / tot


def train(Xtr, ytr, wtr, cfg, seed, epochs, Xva=None, yva=None, wva=None):
    """Mini-batch AdamW with cosine decay. With validation data: early stopping, returns (best loss, best epoch)."""
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    width, depth, wd = cfg
    layers, n = [], Xtr.shape[1]
    for _ in range(depth):
        layers += [torch.nn.Linear(n, width), torch.nn.SiLU()]
        n = width
    net = torch.nn.Sequential(*layers, torch.nn.Linear(n, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS * math.ceil(len(Xtr) / BATCH))
    gen = torch.Generator().manual_seed(seed)
    X, y, w = (torch.tensor(a, dtype=torch.float32) for a in (Xtr, ytr, wtr / wtr.sum()))
    if Xva is not None:
        Xv, yv, wv = (torch.tensor(a, dtype=torch.float32) for a in (Xva, yva, wva / wva.sum()))
    best, best_ep, bad = math.inf, 0, 0
    for ep in range(1, epochs + 1):
        for i in torch.randperm(len(X), generator=gen).split(BATCH):
            opt.zero_grad()
            loss = (w[i] * (net(X[i]).squeeze(1) - y[i]) ** 2).sum() / w[i].sum()
            loss.backward()
            opt.step()
            sched.step()
        if Xva is not None:
            with torch.no_grad():
                v = float((wv * (net(Xv).squeeze(1) - yv) ** 2).sum())
            if v < best - 1e-9:
                best, best_ep, bad = v, ep, 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if Xva is not None:
        return best, best_ep
    return net


def scale(tr, te, cols):
    mu, sd = tr[cols].mean().values, tr[cols].std(ddof=0).values.copy()
    sd[sd == 0] = 1.0
    ym, ys = tr.y.mean(), tr.y.std(ddof=0)
    f = lambda d: ((d[cols].values - mu) / sd).astype(np.float32)
    return f(tr), f(te), ((tr.y.values - ym) / ys), ((te.y.values - ym) / ys), ym, ys


def task(args):
    target, sname, mat = args
    out = (OUT / "folds" if sname in STEP3 else GP.parent / "mlp_folds") / f"{target}_{sname}_{mat}.json"
    if out.exists():
        return str(out)
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    d = load(target, sname)
    cols = SETS[sname]
    tr, te = d[d.material_id != mat].reset_index(drop=True), d[d.material_id == mat].reset_index(drop=True)
    mats = sorted(tr.material_id.unique())
    rng = np.random.default_rng(20261010)  # fixed inner split
    perm = rng.permutation(len(mats))
    inner = [set(np.array(mats)[perm[k::5]]) for k in range(5)]
    table = []
    for cfg in GRID:
        losses, eps = [], []
        for val in inner:
            a, b = tr[~tr.material_id.isin(val)].reset_index(drop=True), tr[tr.material_id.isin(val)].reset_index(drop=True)
            Xa, Xb, ya, yb, _, _ = scale(a, b, cols)
            v, ep = train(Xa, ya, weights(a), cfg, 0, MAX_EPOCHS, Xb, yb, weights(b))
            losses.append(v); eps.append(ep)
        table.append(dict(width=cfg[0], depth=cfg[1], wd=cfg[2], inner_loss=float(np.mean(losses)), epochs=int(round(np.mean(eps)))))
    best = min(table, key=lambda r: r["inner_loss"])
    cfg, epochs = (best["width"], best["depth"], best["wd"]), max(20, best["epochs"])
    Xtr, Xte, ytr, _, ym, ys = scale(tr, te, cols)
    wtr = weights(tr)
    preds = []
    for seed in range(N_ENS):
        net = train(Xtr, ytr, wtr, cfg, seed, epochs)
        with torch.no_grad():
            preds.append(net(torch.tensor(Xte)).squeeze(1).numpy() * ys + ym + te.offset.values)
    preds = np.array(preds)
    res = dict(target=target, input_set=sname, material_id=mat, chosen=best, grid=table, seconds=time.time() - t0,
               energy_MeV=te.energy_MeV.tolist(), depth_mfp=te.depth_mfp.tolist(), label=te[target].tolist(),
               label_se=te[target + "_se"].tolist(), prediction=np.exp(preds.mean(0)).tolist(),
               ensemble_sd_log=preds.std(0, ddof=1).tolist())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(res)); os.replace(tmp, out)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--only", nargs=3)
    ap.add_argument("--sets", nargs="+", default=STEP3)
    a = ap.parse_args()
    mats = sorted(pd.read_csv(DATA).material_id.unique())
    assert len(mats) == 40
    tasks = [tuple(a.only)] if a.only else [(t, s, m) for t in ("EBF", "EABF") for s in a.sets for m in mats]
    t0 = time.time()
    with Pool(a.jobs) as pool:
        for i, f in enumerate(pool.imap_unordered(task, tasks), 1):
            print(f"{i}/{len(tasks)} {Path(f).name} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
