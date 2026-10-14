#!/usr/bin/env python3
"""Step 6 (AMENDMENT5.md section 4, AMENDMENT5A.md): final fits on the 40 development compositions and frozen
predictions on the held-out input rows. No held-out label is read.

Each fit is the procedure of one outer fold of nn_mlp_lomo.py / nn_xgb_lomo.py applied to all 40 materials.
Writes step6/frozen/<target>_<model>.{pt,json}, step6/frozen/<target>_<model>_pred.csv and, with --collect,
step6/heldout_predictions.csv and step6/frozen/manifest.json (SHA-256 of every frozen file).

Usage: nn_final_fit.py [--jobs 14] ; nn_final_fit.py --collect
"""
import argparse
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nn_mlp_lomo as M  # noqa: E402
import nn_xgb_lomo as X  # noqa: E402

ROOT = M.ROOT
S6 = ROOT / "validation/nn_tests_20261010/step6"
FZ = S6 / "frozen"
ARTICLE_F = ROOT / "validation/highE_extension_20261003/surrogate_17E/frozen/EBF_full_features.joblib"
KEYS = ["material_id", "energy_MeV", "depth_mfp"]
TASKS = [(t, k, s) for t in ("EBF", "EABF") for k, s in
         [("MLP", "DR"), ("MLP", "PAy"), ("MLP", "Y2"), ("MLP", "K1r"), ("tXGB", "P1"), ("tXGB", "Y2"), ("tXGB", "K1r")]]


def frames(target, sname):
    xc = pd.read_csv(S6 / "xcom_features.csv")
    dev = M.load(target, sname).merge(xc[xc.part == "development"][KEYS + ["Zeq_xcom"]], on=KEYS, validate="one_to_one")
    dev["Zeq_epics"] = dev.Zeq_xcom  # AMENDMENT5A.md: XCOM-based Zeq for development and held-out alike
    ho = pd.read_csv(S6 / "heldout_inputs.csv").merge(xc[xc.part == "heldout"].drop(columns="part"), on=KEYS, validate="one_to_one")
    ho["Zeq_epics"] = ho.Zeq_xcom
    ho["log_anchor"] = np.log(ho["anchor_" + target])
    assert len(dev) == 5440 and len(ho) == 2192 and ho[M.SETS[sname]].notna().all().all()
    return dev, ho


def inner_split(d):
    mats = sorted(d.material_id.unique())
    perm = np.random.default_rng(20261010).permutation(len(mats))
    return [set(np.array(mats)[perm[k::5]]) for k in range(5)]


def fit(task):
    target, kind, sname = task
    name = f"{target}_{kind}-{sname}"
    out = FZ / f"{name}_pred.csv"
    if out.exists():
        return name
    t0 = time.time()
    dev, ho = frames(target, sname)
    cols = M.SETS[sname]
    table = []
    if kind == "MLP":
        import torch
        torch.set_num_threads(1)
        for cfg in M.GRID:
            losses, eps = [], []
            for val in inner_split(dev):
                a, b = dev[~dev.material_id.isin(val)].reset_index(drop=True), dev[dev.material_id.isin(val)].reset_index(drop=True)
                Xa, Xb, ya, yb, _, _ = M.scale(a, b, cols)
                v, ep = M.train(Xa, ya, M.weights(a), cfg, 0, M.MAX_EPOCHS, Xb, yb, M.weights(b))
                losses.append(v); eps.append(ep)
            table.append(dict(width=cfg[0], depth=cfg[1], wd=cfg[2], inner_loss=float(np.mean(losses)), epochs=int(round(np.mean(eps)))))
        best = min(table, key=lambda r: r["inner_loss"])
        cfg, epochs = (best["width"], best["depth"], best["wd"]), max(20, best["epochs"])
        mu, sd = dev[cols].mean().values, dev[cols].std(ddof=0).values.copy()
        sd[sd == 0] = 1.0
        ym, ys = float(dev.y.mean()), float(dev.y.std(ddof=0))
        Xd = ((dev[cols].values - mu) / sd).astype(np.float32)
        Xh = torch.tensor(((ho[cols].values - mu) / sd).astype(np.float32))
        yd, wd_ = (dev.y.values - ym) / ys, M.weights(dev)
        preds, states = [], []
        for seed in range(M.N_ENS):
            net = M.train(Xd, yd, wd_, cfg, seed, epochs)
            states.append(net.state_dict())
            with torch.no_grad():
                preds.append(net(Xh).squeeze(1).numpy() * ys + ym)
        preds = np.array(preds)
        logp, sdl = preds.mean(0), preds.std(0, ddof=1)
        torch.save(dict(columns=cols, config=cfg, epochs=epochs, x_mean=mu, x_sd=sd, y_mean=ym, y_sd=ys, states=states), FZ / f"{name}.pt")
    else:
        for cfg in X.GRID:
            losses, its = [], []
            for val in inner_split(dev):
                a, b = dev[~dev.material_id.isin(val)], dev[dev.material_id.isin(val)]
                wb = X.wts(b)
                m = X.model(cfg, 1500, 50).fit(a[cols], a.y, sample_weight=X.wts(a), eval_set=[(b[cols], b.y)],
                                               sample_weight_eval_set=[wb], verbose=False)
                p = m.predict(b[cols])
                losses.append(float((wb * (p - b.y.values) ** 2).sum() / wb.sum())); its.append(m.best_iteration + 1)
            table.append(dict(max_depth=cfg[0], learning_rate=cfg[1], min_child_weight=cfg[2],
                              inner_loss=float(np.mean(losses)), trees=int(round(np.mean(its)))))
        best = min(table, key=lambda r: r["inner_loss"])
        m = X.model((best["max_depth"], best["learning_rate"], best["min_child_weight"]), max(20, best["trees"]))
        m.fit(dev[cols], dev.y, sample_weight=X.wts(dev))
        m.save_model(FZ / f"{name}.json")
        logp, sdl = m.predict(ho[cols]).astype(float), np.full(len(ho), np.nan)
    (FZ / f"{name}_selection.json").write_text(json.dumps(dict(target=target, model=f"{kind}-{sname}", columns=cols, chosen=best, grid=table,
                                                               seconds=time.time() - t0), indent=1) + "\n")
    ho[["cohort"] + KEYS].assign(target=target, model=f"{kind}-{sname}", prediction=np.exp(logp), ensemble_sd_log=sdl).to_csv(out, index=False)
    return name


def collect():
    import joblib
    parts = [pd.read_csv(FZ / f"{t}_{k}-{s}_pred.csv") for t, k, s in TASKS]
    ho = pd.read_csv(S6 / "heldout_inputs.csv")
    art = joblib.load(ARTICLE_F)
    art = art["model"] if isinstance(art, dict) else art
    cols = list(art.get_booster().feature_names)
    x = ho.assign(log_anchor=np.log(ho.anchor_EBF))
    parts.append(ho[["cohort"] + KEYS].assign(target="EBF", model="XGBoost-F", prediction=np.maximum(1.0, np.exp(art.predict(x[cols]).astype(float))), ensemble_sd_log=np.nan))
    for t in ("EBF", "EABF"):
        parts.append(ho[["cohort"] + KEYS].assign(target=t, model="Reference alone", prediction=ho["anchor_" + t], ensemble_sd_log=np.nan))
    out = pd.concat(parts, ignore_index=True)
    assert out.prediction.notna().all() and (out.groupby(["target", "model"]).size() == 2192).all()
    out.to_csv(S6 / "heldout_predictions.csv", index=False)
    files = sorted(p for p in FZ.iterdir() if p.name != "manifest.json") + [S6 / "heldout_inputs.csv", S6 / "xcom_features.csv",
                                                                             S6 / "heldout_predictions.csv", ARTICLE_F]
    man = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (FZ / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(out.groupby(["target", "model"]).prediction.agg(["min", "max"]).round(3))
    print(len(man), "files hashed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    FZ.mkdir(parents=True, exist_ok=True)
    if a.collect:
        return collect()
    t0 = time.time()
    with Pool(a.jobs) as pool:
        for n in pool.imap_unordered(fit, TASKS):
            print(n, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
