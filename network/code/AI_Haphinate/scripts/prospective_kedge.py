#!/usr/bin/env python3
"""Prospective K-edge test of the frozen networks (validation/prospective_kedge_20261011/REGISTRATION.md).

No model is fitted. The frozen files of validation/nn_tests_20261010/step6 predict twelve new compositions whose
transport labels do not exist when the predictions are hashed.

  select    seeded choice of the twelve compositions from label-free files (run once)
  features  response tables, 100 keV fluorescence reference and atomic descriptors, input rows
  predict   predictions of every frozen model, empirical interval widths, self-check on the earlier cohort
  register  macros, seed plan, registration.json with SHA-256 of every frozen file
  verify    recompute the registered hashes
  run       Geant4 labels (BF), fixed seeds, resumable
  evaluate  the single evaluation (refuses to run twice)

The XCOM-based inputs of the published-style sets are written by scripts/prospective_kedge_xcom.py (system python3)
between features and predict.
"""
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BF = ROOT.parent / "BF"
V = ROOT / "validation"
OUT = V / "prospective_kedge_20261011"
FZ = OUT / "frozen"
S6 = V / "nn_tests_20261010/step6"
FR = V / "physics_surrogate_full_range_20260930"
RUNS = BF / "runs/prospective_kedge_20261011"
KEYS = ["material_id", "energy_MeV", "depth_mfp"]

SELECTION_SEED = 20261011
MATCH_TOL = 0.0015  # relative difference of the Compton-to-total ratio at 0.1 MeV within a pair
ABS = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy"]
TRA = ["Er", "Tm", "Yb", "Lu", "Y"]
ENDMEMBER = dict(La="LAH", Ce="CEH", Pr="PRH", Nd="NDH", Sm="SMH", Eu="EUH", Gd="GDH", Tb="TBH", Dy="DYH", Ho="HOH",
                 Er="ERH", Tm="TMH", Yb="YBH", Lu="LUH", Y="YH")
ELEMENTS = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Y", "Yb", "Lu", "Sc", "Hf", "O"]

ENERGIES = [0.1, 0.1225, 0.15]
DEPTHS = [0.5, 1.0, 2.0, 3.0, 4.0]
# seed tiers: (first index, one past the last index, depths scored by each seed process)
TIERS = {0.1: [(0, 64, [0.5, 1.0, 2.0, 3.0, 4.0]), (64, 256, [3.0, 4.0]), (256, 640, [4.0])],
         0.1225: [(0, 64, [0.5, 1.0, 2.0, 3.0, 4.0])],
         0.15: [(0, 64, [0.5, 1.0, 2.0, 3.0, 4.0])]}
EVENTS, CUT_MM = 10000, 0.01
NB, BOOT_SEED, MARGIN, GATE = 10000, 20261012, 0.25, 5.0
MODELS_MLP = ["DR", "PAy", "Y2", "K1r"]
MODELS_XGB = ["P1", "Y2", "K1r"]
TREE = {"EBF": "XGBoost-F", "EABF": "tXGB-P1"}
ARTICLE_F = V / "highE_extension_20261003/surrogate_17E/frozen/EBF_full_features.joblib"
LABEL_BINARY = BF / "build_deep_validation/bin/BF"
LABEL_BINARY_SHA = "a9b49791bfba73133808f690159f030fc1b4143494acc262ad38ca9d981d7fde"
G4DATA = ROOT.parent / "geant4/install/share/Geant4/data/G4EMLOW8.8"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- selection

def aw(symbol):
    import xraylib as xr
    return xr.AtomicWeight(xr.SymbolToAtomicNumber(symbol))


def mass_fractions(recipe):
    mass = {e: 2 * x * aw(e) for e, x in recipe.items()}
    mass.update(Hf=2 * aw("Hf"), O=7 * aw("O"))
    total = sum(mass.values())
    return {e: mass.get(e, 0.0) / total for e in ELEMENTS}


def known_compositions():
    """Mass-fraction vectors and cation sets of every composition simulated before this test."""
    rows = []
    cat = pd.read_csv(ROOT / "materials/materials.csv")
    for r in cat.itertuples():
        rows.append((r.material_id, {e: float(getattr(r, e + "_wt_fraction")) for e in ELEMENTS}))
    for f in [FR / "fresh_confirmation/frozen/recipes.json", FR / "fresh_confirmation_amendment1/new6/frozen/recipes.json"]:
        rows += [(m["material_id"], {e: float(m["element_mass_fractions"][e]) for e in ELEMENTS}) for m in json.loads(f.read_text())]
    for f in [V / "blind_esgyp_20261008/frozen/materials.csv", V / "blind_esgyp_20261008/amendment1/frozen_lnttl/materials.csv"]:
        for r in pd.read_csv(f).itertuples():
            rows.append((r.material_id, {e: float(getattr(r, e + "_wt_fraction")) for e in ELEMENTS}))
    assert len(rows) == 72 and len({m for m, _ in rows}) == 72
    sets = {m: frozenset(e for e in ELEMENTS if w[e] > 0 and e not in ("Hf", "O")) for m, w in rows}
    return rows, sets


def select():
    sel = OUT / "selection"
    if (sel / "selected.json").exists():
        raise SystemExit("selection/selected.json exists: the selection is made once")
    sel.mkdir(parents=True, exist_ok=True)
    known, known_sets = known_compositions()
    used_sets = set(known_sets.values())
    kvec = np.array([[w[e] for e in ELEMENTS] for _, w in known])
    log = [f"selection seed {SELECTION_SEED}; numpy default_rng; {len(known)} known compositions; {len(used_sets)} used cation sets"]

    def distance(recipe):
        v = np.array([mass_fractions(recipe)[e] for e in ELEMENTS])
        return float(np.abs(kvec - v).sum(1).min())

    pool = pd.read_csv(V / "kedge_test_20260928/candidates.csv")
    pool["recipe"] = pool.A.map(lambda s: {p.split(":")[0]: float(p.split(":")[1]) for p in s.split(";")})
    pool["cset"] = pool.recipe.map(frozenset)
    pool["dist"] = pool.recipe.map(distance)
    absorbers = pool[(pool.cls == "absorber") & ~pool.cset.isin(used_sets)].reset_index(drop=True)
    transmitters = pool[(pool.cls == "transmitter") & (pool.dist > 0.02)].reset_index(drop=True)
    log.append(f"pool {len(pool)}: absorbers with an unused cation set {len(absorbers)}; transmitters farther than 0.02 from every known composition {len(transmitters)}")
    eligible = [(a.A, t.A) for a in absorbers.itertuples() for t in transmitters.itertuples()
                if abs(a.R - t.R) / (0.5 * (a.R + t.R)) <= MATCH_TOL]
    eligible.sort()
    log.append(f"eligible matched pairs {len(eligible)}")
    rng = np.random.default_rng(SELECTION_SEED)
    order = rng.permutation(len(eligible))
    rec = {r.A: r for r in pool.itertuples()}
    pairs, used_abs_sets, used_tra, used_heavy = [], set(), set(), set()
    for strict in (True, False):
        for i in order:
            a, t = eligible[i]
            heavy = next((e for e in rec[t].recipe if e != "Y"), "Y")
            if len(pairs) == 4 or rec[a].cset in used_abs_sets or t in used_tra or (strict and heavy in used_heavy):
                continue
            pairs.append((a, t)); used_abs_sets.add(rec[a].cset); used_tra.add(t); used_heavy.add(heavy)
        log.append(f"after the {'first' if strict else 'second'} pass: {len(pairs)} pairs")
        if len(pairs) == 4:
            break
    assert len(pairs) == 4
    chosen = []
    for k, (a, t) in enumerate(pairs, 1):
        for role, s in (("absorber", a), ("transmitter", t)):
            chosen.append(dict(material_id=f"PK_P{k}{role[0].upper()}", group="pair", pair=f"Q{k}", role=role,
                               recipe=rec[s].recipe, R_0p1_MeV=float(rec[s].R)))
    taken = used_sets | {frozenset(c["recipe"]) for c in chosen}

    # two mixed-class binaries: one absorber and one transmitter rare earth on the same site
    mixed_pool = sorted((a, t) for a in ABS for t in TRA if frozenset((a, t)) not in taken)
    log.append(f"mixed-class cation pairs with an unused set: {len(mixed_pool)}")
    mixed, seen = [], set()
    for i in rng.permutation(len(mixed_pool)):
        a, t = mixed_pool[i]
        if len(mixed) == 2 or a in seen or t in seen:
            continue
        mixed.append((a, t)); seen |= {a, t}
    for k, (a, t) in enumerate(mixed, 1):
        x = float(rng.choice([0.3, 0.4, 0.5, 0.6, 0.7]))
        chosen.append(dict(material_id=f"PK_M{k}", group="mixed", pair="", role="mixed", recipe={a: x, t: round(1 - x, 1)}, R_0p1_MeV=None))

    # two Ho-rich binaries: the first with an absorber partner, the second with a transmitter partner
    for k, partners in enumerate((ABS, TRA), 1):
        options = sorted(p for p in partners if frozenset(("Ho", p)) not in taken)
        p = str(rng.choice(options))
        x = float(rng.choice([0.5, 0.6, 0.7, 0.8]))
        chosen.append(dict(material_id=f"PK_H{k}", group="Ho-rich", pair="", role="Ho with " + ("absorber" if k == 1 else "transmitter"),
                           recipe={"Ho": x, p: round(1 - x, 1)}, R_0p1_MeV=None))

    cat = pd.read_csv(ROOT / "materials/materials.csv").set_index("material_id")
    for c in chosen:
        w = mass_fractions(c["recipe"])
        total = sum(2 * x * aw(e) for e, x in c["recipe"].items()) + 2 * aw("Hf") + 7 * aw("O")
        component = {e: x * (2 * aw(e) + 2 * aw("Hf") + 7 * aw("O")) / total for e, x in c["recipe"].items()}
        c["density_g_cm3"] = 1 / sum(f / float(cat.loc[ENDMEMBER[e], "density_g_cm3"]) for e, f in component.items())
        c["element_mass_fractions"] = w
        c["formula"] = "(" + "".join(f"{e}{x:g}" for e, x in c["recipe"].items()) + ")2Hf2O7"
        d = np.abs(kvec - np.array([w[e] for e in ELEMENTS])).sum(1)
        c["nearest_known"] = known[int(d.argmin())][0]
        c["nearest_known_L1"] = float(d.min())
        assert c["nearest_known_L1"] > 0.02, c
    assert len({frozenset(c["recipe"].items()) for c in chosen}) == 12
    pool.drop(columns=["recipe", "cset"]).to_csv(sel / "pool_with_distance.csv", index=False)
    pd.DataFrame(eligible, columns=["absorber", "transmitter"]).to_csv(sel / "eligible_pairs.csv", index=False)
    save_json(sel / "selected.json", dict(seed=SELECTION_SEED, match_tolerance=MATCH_TOL, materials=chosen))
    (sel / "SELECTION_LOG.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    print(pd.DataFrame(chosen)[["material_id", "group", "pair", "role", "formula", "density_g_cm3", "R_0p1_MeV", "nearest_known", "nearest_known_L1"]].to_string(index=False))


# ---------------------------------------------------------------- features (label-free)

def response_table(RT, tables, weights, path):
    """Edge-preserving additive NIST table, the construction of the frozen fresh test and of the retrospective cohorts."""
    w = {e: x for e, x in weights.items() if x}
    knots = sorted({r[0] for e in w for r in tables[e]} | {.6617, 1.25, 2.})
    edges = {x for e in w for x, n in Counter(r[0] for r in tables[e]).items() if n == 2}
    grid = sorted(set(knots) | {a * (b / a) ** (j / 8) for a, b in zip(knots, knots[1:]) for j in range(1, 8)})
    values = []
    for en in grid:
        if en in edges:
            values.append((en, RT.mixture(tables, w, en, "left")))
        values.append((en, RT.mixture(tables, w, en)))
    RT.write_csv(path, ["Energy_MeV", "mu_en_rho_cm2_g"], values)


def features(a):
    sys.path.insert(0, str(ROOT / "scripts"))
    import confirm_physics_full_range as CP
    from nn_mlp_lomo import ATOMIC, WT
    if (FZ / "inputs.csv").exists():
        raise SystemExit("frozen/inputs.csv exists")
    FZ.mkdir(exist_ok=True)
    chosen = json.loads((OUT / "selection/selected.json").read_text())["materials"]
    rows = [dict(material_id=c["material_id"], density_g_cm3=c["density_g_cm3"], **{e + "_wt_fraction": c["element_mass_fractions"][e] for e in ELEMENTS})
            for c in chosen]
    pd.DataFrame(rows).to_csv(FZ / "materials.csv", index=False)
    weights = {c["material_id"]: c["element_mass_fractions"] for c in chosen}
    cat = pd.read_csv(ROOT / "materials/materials.csv").set_index("material_id")
    check_ids = ["KT_LAND", "KT_TMY6"]  # pipeline check against the features frozen for the earlier K-edge cohort
    for m in check_ids:
        weights[m] = {e: float(cat.loc[m, e + "_wt_fraction"]) for e in ELEMENTS}
    raw = V / "response_20260928/raw"
    tables = {el: CP.RT.parse_table((raw / f"z{CP.RT.Z[el]:02d}.html").read_text()) for el in sorted({e for w in weights.values() for e, x in w.items() if x})}
    response = FZ / "response_tables"; response.mkdir(exist_ok=True)
    shutil.copy2(FR / "fresh_confirmation/frozen/reference_air.csv", FZ / "reference_air.csv")
    scratch = OUT / "pipeline_check"; scratch.mkdir(exist_ok=True)
    for m, w in weights.items():
        response_table(CP.RT, tables, w, (scratch if m in check_ids else response) / f"{m}.csv")
    tasks = [(m, w, str(scratch if m in check_ids else response), str(FZ / "reference_air.csv"), DEPTHS) for m, w in weights.items()]
    out, diag = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for m, (vals, d) in zip(weights, pool.map(CP.anchor_task, tasks)):
            out.extend(vals); diag.append(d); print("reference", m, f"{d['wall_s']:.0f}s", flush=True)
    phys = pd.DataFrame(out).drop(columns="energy_MeV").rename(columns={"physics_EBF": "anchor_EBF", "physics_EABF": "anchor_EABF"})
    old = pd.read_csv(FR / "retrospective_existing/physical_features.csv")
    chk = phys[phys.material_id.isin(check_ids)].merge(old, on=["material_id", "depth_mfp"], suffixes=("", "_old"), validate="one_to_one")
    assert len(chk) == 10
    worst = max(float(np.abs(chk[c] / chk[c + "_old"] - 1).max()) for c in ["anchor_EBF", "anchor_EABF"] + ATOMIC)
    assert worst < 1e-6, worst
    phys = phys[~phys.material_id.isin(check_ids)]
    phys.to_csv(FZ / "physical_features.csv", index=False)
    save_json(FZ / "anchor_diagnostics.json", dict(settings=diag, pipeline_check=dict(materials=check_ids, worst_relative_difference=worst)))
    coords = pd.DataFrame([dict(cohort="prospective", material_id=c["material_id"], energy_MeV=e, depth_mfp=d, log10_E_MeV=np.log10(e),
                                density_g_cm3=c["density_g_cm3"], **{el + "_wt": c["element_mass_fractions"][el] for el in ELEMENTS})
                           for c in chosen for e in ENERGIES for d in DEPTHS])
    frame = coords.merge(phys, on=["material_id", "depth_mfp"], validate="many_to_one")
    cols = ["cohort"] + KEYS + ["log10_E_MeV", "density_g_cm3"] + WT + ["anchor_EBF", "anchor_EABF"] + ATOMIC
    assert len(frame) == 180 and frame[cols].notna().all().all()
    frame[cols].to_csv(FZ / "inputs.csv", index=False)
    dev = pd.read_csv(V / "highE_extension_20261003/surrogate_17E/training_data.csv")
    rng_ = frame[["density_g_cm3", "anchor_EBF", "anchor_EABF"] + ATOMIC].agg(["min", "max"]).T.join(
        dev[["density_g_cm3", "anchor_EBF", "anchor_EABF"] + ATOMIC].agg(["min", "max"]).T, rsuffix="_dev")
    rng_["inside_dev_range"] = (rng_["min"] >= rng_["min_dev"]) & (rng_["max"] <= rng_["max_dev"])
    rng_.to_csv(FZ / "input_ranges.csv")
    print(rng_.round(4).to_string()); print("pipeline check worst relative difference", worst)


# ---------------------------------------------------------------- predictions of the frozen models

def step6_manifest_ok():
    man = json.loads((S6 / "frozen/manifest.json").read_text())
    bad = [f for f, h in man.items() if digest(ROOT / f) != h]
    assert not bad, bad
    return man


def predict_all(ho):
    """ho: input rows merged with the XCOM features. Returns long predictions of every frozen model."""
    import joblib
    import torch
    import xgboost
    torch.set_num_threads(1)
    parts = []
    base = ho[["cohort"] + KEYS]
    for t in ("EBF", "EABF"):
        x = ho.assign(Zeq_epics=ho.Zeq_xcom, log_anchor=np.log(ho["anchor_" + t]))
        for s in MODELS_MLP:
            f = torch.load(S6 / f"frozen/{t}_MLP-{s}.pt", weights_only=False)
            width, depth, _ = f["config"]
            X = torch.tensor(((x[f["columns"]].values - f["x_mean"]) / f["x_sd"]).astype(np.float32))
            preds = []
            for state in f["states"]:
                layers, n = [], len(f["columns"])
                for _ in range(depth):
                    layers += [torch.nn.Linear(n, width), torch.nn.SiLU()]
                    n = width
                net = torch.nn.Sequential(*layers, torch.nn.Linear(n, 1))
                net.load_state_dict(state)
                with torch.no_grad():
                    preds.append(net(X).squeeze(1).numpy() * f["y_sd"] + f["y_mean"])
            parts.append(base.assign(target=t, model=f"MLP-{s}", prediction=np.exp(np.mean(preds, axis=0))))
        for s in MODELS_XGB:
            cols = json.loads((S6 / f"frozen/{t}_tXGB-{s}_selection.json").read_text())["columns"]
            m = xgboost.XGBRegressor(); m.load_model(S6 / f"frozen/{t}_tXGB-{s}.json")
            parts.append(base.assign(target=t, model=f"tXGB-{s}", prediction=np.exp(m.predict(x[cols]).astype(float))))
        parts.append(base.assign(target=t, model="Reference alone", prediction=ho["anchor_" + t].values))
    art = joblib.load(ARTICLE_F)
    art = art["model"] if isinstance(art, dict) else art
    x = ho.assign(log_anchor=np.log(ho.anchor_EBF))
    parts.append(base.assign(target="EBF", model="XGBoost-F",
                             prediction=np.maximum(1.0, np.exp(art.predict(x[list(art.get_booster().feature_names)]).astype(float)))))
    return pd.concat(parts, ignore_index=True)


def predict():
    if (FZ / "predictions.csv").exists():
        raise SystemExit("frozen/predictions.csv exists")
    step6_manifest_ok()
    # self-check: this routine reproduces the predictions frozen for the earlier K-edge cohort
    xc = pd.read_csv(S6 / "xcom_features.csv")
    old = pd.read_csv(S6 / "heldout_inputs.csv").query("cohort == 'kedge'").merge(xc[xc.part == "heldout"].drop(columns="part"), on=KEYS, validate="one_to_one")
    ref = pd.read_csv(S6 / "heldout_predictions.csv").query("cohort == 'kedge'")
    cmp_ = predict_all(old).merge(ref, on=["cohort"] + KEYS + ["target", "model"], suffixes=("", "_frozen"), validate="one_to_one")
    assert len(cmp_) == len(ref) == 120 * 17, (len(cmp_), len(ref))
    worst = float(np.abs(cmp_.prediction / cmp_.prediction_frozen - 1).max())
    assert worst < 1e-5, worst
    ho = pd.read_csv(FZ / "inputs.csv").merge(pd.read_csv(FZ / "xcom_features.csv"), on=KEYS, validate="one_to_one")
    assert len(ho) == 180
    pred = predict_all(ho)
    # empirical 90 % interval half-widths in ln B from the development out-of-fold scores (Amendment 5, section 7.5)
    sys.path.insert(0, str(ROOT / "scripts"))
    import nn_heldout_evaluate as HE
    q = HE.dev_scores()
    rows = []
    for (t, m), g in q.groupby(["target", "model"]):
        g = g.set_index(["energy_MeV", "depth_mfp"]).score
        for d in DEPTHS:
            lo, hi = float(g[(0.1, d)]), float(g[(0.15, d)])
            f = np.log(0.1225 / 0.1) / np.log(0.15 / 0.1)
            rows += [dict(target=t, model=m, energy_MeV=e, depth_mfp=d, score=s) for e, s in ((0.1, lo), (0.1225, lo + f * (hi - lo)), (0.15, hi))]
    pred = pred.merge(pd.DataFrame(rows), on=["target", "model", "energy_MeV", "depth_mfp"], how="left", validate="many_to_one")
    assert pred.prediction.notna().all() and (pred.groupby(["target", "model"]).size() == 180).all()
    pred.to_csv(FZ / "predictions.csv", index=False)
    save_json(FZ / "predict_selfcheck.json", dict(rows=len(cmp_), worst_relative_difference_from_frozen_kedge_predictions=worst))
    print("self-check worst relative difference", worst)
    print(pred[np.isclose(pred.energy_MeV, 0.1) & (pred.depth_mfp == 4.0) & pred.model.isin(["MLP-DR", "MLP-Y2", "MLP-K1r"])]
          .pivot_table(index="material_id", columns=["target", "model"], values="prediction").round(3).to_string())


# ---------------------------------------------------------------- registration and run

def plan_rows(driver):
    mats = list(csv.DictReader((FZ / "materials.csv").open()))
    rows = []
    for row in mats:
        for e in ENERGIES:
            for first, last, depths in TIERS[e]:
                for i in range(first, last):
                    pair = driver.seed_pair(row["material_id"], f"{e:g}", i)
                    text = driver.material_macro(row, f"{e:g}", [f"{d:g}" for d in depths], EVENTS, pair, CUT_MM, None)
                    rows.append(dict(material_id=row["material_id"], energy_MeV=e, index=i, seed1=pair[0], seed2=pair[1],
                                     depths=",".join(f"{d:g}" for d in depths), macro_sha256=hashlib.sha256(text.encode()).hexdigest()))
    return mats, rows


def tracked_files():
    files = sorted(p for p in FZ.rglob("*") if p.is_file()) + sorted((OUT / "selection").glob("*")) + [OUT / "REGISTRATION.md",
             Path(__file__), ROOT / "scripts/prospective_kedge_xcom.py", S6 / "frozen/manifest.json", V / "kedge_test_20260928/candidates.csv"]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def register():
    if (OUT / "registration.json").exists():
        raise SystemExit("registration.json exists")
    if RUNS.exists():
        raise SystemExit(f"{RUNS} exists before registration")
    assert digest(LABEL_BINARY) == LABEL_BINARY_SHA
    shutil.copy2(BF / "scripts/generate_labels.py", FZ / "bf_driver.py")
    driver = load_module(FZ / "bf_driver.py", "pk_driver")
    mats, rows = plan_rows(driver)
    pd.DataFrame(rows).to_csv(FZ / "seed_plan.csv", index=False)
    beamons = sum(len(r["depths"].split(",")) for r in rows)
    atomic = {str(p.relative_to(G4DATA)): digest(p) for p in sorted(G4DATA.rglob("*")) if p.is_file()}
    save_json(FZ / "g4emlow_sha256.json", atomic)
    reg = dict(status="prospective registration; no transport label of these compositions exists",
               registered_utc=datetime.now(timezone.utc).isoformat(),
               materials=[m["material_id"] for m in mats], energies_MeV=ENERGIES, depths_mfp=DEPTHS,
               seed_tiers={f"{e:g}": [dict(first=a, last_exclusive=b, depths=d) for a, b, d in t] for e, t in TIERS.items()},
               events_per_seed_per_depth=EVENTS, seed_processes=len(rows), depth_runs=beamons, histories=beamons * EVENTS,
               label_binary_sha256=LABEL_BINARY_SHA, G4LEDATA=str(G4DATA.resolve()), cut_mm=CUT_MM,
               primary=dict(energy_MeV=0.1, ours="MLP-DR", comparators=["MLP-Y2", "MLP-K1r"], targets=["EBF", "EABF"], margin_points=MARGIN,
                            bootstrap=dict(resamples=NB, seed=BOOT_SEED, design="pairs resampled as blocks; mixed and Ho-rich compositions within their strata")),
               gate_percent=GATE, sha256=tracked_files())
    save_json(OUT / "registration.json", reg)
    (OUT / "registration.sha256").write_text(digest(OUT / "registration.json") + "\n")
    print(f"registered: {len(rows)} seed processes, {beamons} depth runs, {beamons * EVENTS:,} histories; {len(reg['sha256'])} files hashed")


def verify():
    reg = json.loads((OUT / "registration.json").read_text())
    assert digest(OUT / "registration.json") == (OUT / "registration.sha256").read_text().strip(), "registration changed"
    bad = [f for f, h in reg["sha256"].items() if digest(ROOT / f) != h]
    assert not bad, bad
    step6_manifest_ok()
    return reg


def run(a):
    reg = verify()
    driver = load_module(FZ / "bf_driver.py", "pk_driver_run")
    mats = {r["material_id"]: r for r in csv.DictReader((FZ / "materials.csv").open())}
    plan = pd.read_csv(FZ / "seed_plan.csv", dtype={"depths": str})
    RUNS.mkdir(parents=True, exist_ok=True)
    binary = RUNS / "BF_snapshot"
    if not binary.exists():
        shutil.copy2(LABEL_BINARY, binary)
    assert digest(binary) == reg["label_binary_sha256"]
    os.environ["G4LEDATA"] = reg["G4LEDATA"]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("BF_", "NIST_"))}
    env.update(BF_FIXED_EVENTS="1", BF_VR_MODE="off", BF_RESPONSE_TABLE_DIR=str((FZ / "response_tables").resolve()),
               BF_RESPONSE_REQUIRE_TABLE="1", BF_RESPONSE_BOUND="lower")
    if not (RUNS / "run_started.json").exists():
        save_json(RUNS / "run_started.json", dict(started_utc=datetime.now(timezone.utc).isoformat(), jobs=a.jobs,
                                                  registration_sha256=digest(OUT / "registration.json")))

    def one(r):
        e = f"{r.energy_MeV:g}"
        args = SimpleNamespace(output=RUNS / "seeds", mfps=r.depths.split(","), events=EVENTS, cut_mm=CUT_MM, shell_fraction=None)
        folder = args.output / r.material_id / f"E{e}" / f"seed_{r.index:03d}"
        done = (folder / "complete.json").exists()
        t0 = time.perf_counter()
        try:
            driver.run_seed(args, binary, env, mats[r.material_id], e, r.index)
        except Exception as err:  # rerun later with the identical macro; nothing is dropped
            return f"FAILED {folder}: {err}"
        assert digest(folder / "m.txt") == r.macro_sha256, folder
        if not done:
            save_json(folder / "runtime.json", dict(wall_seconds=time.perf_counter() - t0))
        return None

    order = plan.assign(n=plan.depths.str.count(",")).sort_values(["n", "material_id", "energy_MeV", "index"], ascending=[True, True, True, True])
    for attempt in range(3):
        t0 = time.time(); failed = []
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            for k, msg in enumerate(pool.map(one, list(order.itertuples(index=False))), 1):
                if msg:
                    failed.append(msg); print(msg, flush=True)
                if k % 500 == 0:
                    print(f"{k}/{len(order)} seed processes, {time.time() - t0:.0f} s", flush=True)
        if not failed:
            break
        print(f"attempt {attempt + 1}: {len(failed)} failed seed processes; rerunning", flush=True)
    if failed:
        raise SystemExit("seed processes still failing; not complete")
    verify()
    save_json(RUNS / "run_completed.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(), seed_processes=len(order)))
    print("ALL_DONE", flush=True)

# ---------------------------------------------------------------- the single evaluation

def md(t):
    t = t.copy()
    for c in t.columns:
        if t[c].dtype.kind == "f":
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    return "\n".join(["| " + " | ".join(map(str, t.columns)) + " |", "|" + "---|" * len(t.columns)] + ["| " + " | ".join(map(str, r)) + " |" for r in t.values])


def read_labels(plan):
    """Mean and standard error over independent seeds for every point; every planned seed process must be complete."""
    driver = load_module(FZ / "bf_driver.py", "pk_driver_eval")
    seeds = []
    for r in plan.itertuples(index=False):
        folder = RUNS / "seeds" / r.material_id / f"E{r.energy_MeV:g}" / f"seed_{r.index:03d}"
        marker = json.loads((folder / "complete.json").read_text())
        for name, sha in marker["files"].items():
            assert digest(folder / name) == sha, folder
        assert digest(folder / "m.txt") == r.macro_sha256 and marker["seeds"] == driver.seed_pair(r.material_id, f"{r.energy_MeV:g}", r.index)
        depths = r.depths.split(",")
        assert [f"{float(x['MFP']):g}" for x in marker["rows"]] == depths, folder
        for x in marker["rows"]:
            seeds.append(dict(material_id=r.material_id, energy_MeV=r.energy_MeV, depth_mfp=float(x["MFP"]), index=r.index,
                              EBF=float(x["EBF_TL"]), EABF=float(x["EABF_TL"]), EBF_crossing=float(x["EBF_air"]), EABF_crossing=float(x["EABF_mat"]),
                              transport_ratio=float(x["TransportRatio"])))
    seeds = pd.DataFrame(seeds)
    rows = []
    for key, g in seeds.groupby(KEYS):
        g = g.sort_values("index"); n = len(g)
        row = dict(zip(KEYS, key), seeds=n, transport_ratio=g.transport_ratio.mean(), transport_ratio_se=g.transport_ratio.std(ddof=1) / np.sqrt(n))
        for t in ("EBF", "EABF"):
            v = g[t].values
            row.update({t: v.mean(), t + "_se": v.std(ddof=1) / np.sqrt(n), t + "_crossing": g[t + "_crossing"].mean(), t + "_max_seed_share": v.max() / v.sum(),
                        t + "_largest_seed_over_mean": v.max() / v.mean(), t + "_mean_first_quarter": v[: n // 4].mean(), t + "_mean_first_half": v[: n // 2].mean(),
                        t + "_se_first_half": v[: n // 2].std(ddof=1) / np.sqrt(n // 2)})
        rows.append(row)
    return seeds, pd.DataFrame(rows)


def strat_boot(diff, groups, rng):
    """diff: per-composition values indexed by material_id. Pairs are resampled as blocks, the others within their stratum."""
    pairs = sorted({g["pair"] for g in groups.values() if g["group"] == "pair"})
    blocks = np.array([[diff[m] for m, g in groups.items() if g["pair"] == q] for q in pairs])  # 4 x 2
    out = blocks[rng.integers(0, len(pairs), (NB, len(pairs)))].sum((1, 2))
    for stratum in ("mixed", "Ho-rich"):
        v = np.array([diff[m] for m, g in groups.items() if g["group"] == stratum])
        out = out + v[rng.integers(0, len(v), (NB, len(v)))].sum(1)
    return out / len(diff)


def evaluate():
    if (OUT / "evaluation.json").exists():
        raise SystemExit("evaluation.json exists: the evaluation is run once")
    reg = verify()
    assert (RUNS / "run_completed.json").exists(), "run not complete"
    plan = pd.read_csv(FZ / "seed_plan.csv", dtype={"depths": str})
    seeds, lab = read_labels(plan)
    expect = {(0.1, 0.5): 64, (0.1, 1.0): 64, (0.1, 2.0): 64, (0.1, 3.0): 256, (0.1, 4.0): 640}
    assert len(lab) == 180 and all(r.seeds == expect.get((r.energy_MeV, r.depth_mfp), 64) for r in lab.itertuples())
    seeds.to_csv(OUT / "seed_values.csv.gz", index=False); lab.to_csv(OUT / "labels.csv", index=False)
    groups = {c["material_id"]: c for c in json.loads((OUT / "selection/selected.json").read_text())["materials"]}
    long = pd.concat([lab[KEYS + [t, t + "_se"]].rename(columns={t: "label", t + "_se": "label_se"}).assign(target=t) for t in ("EBF", "EABF")])
    j = pd.read_csv(FZ / "predictions.csv").merge(long, on=KEYS + ["target"], validate="many_to_one")
    assert len(j) == 180 * 17 and np.isfinite(j.label).all() and (j.label > 0).all()
    j["group"] = j.material_id.map(lambda m: groups[m]["group"]); j["role"] = j.material_id.map(lambda m: groups[m]["role"])
    j["res"] = 100 * (j.prediction / j.label - 1); j["ape"] = j.res.abs(); j["se_pct"] = 100 * j.label_se / j.label
    j["z"] = j.res / (j.se_pct * j.prediction / j.label)
    j["covered"] = np.where(j.score.notna(), np.abs(np.log(j.prediction / j.label)) <= j.score, np.nan)
    j.to_csv(OUT / "joined.csv", index=False)
    rng = np.random.default_rng(BOOT_SEED)
    at = lambda e: np.isclose(j.energy_MeV, e)

    def comp_err(model, target, e):
        d = j[(j.model == model) & (j.target == target) & at(e)]
        return d.groupby("material_id").ape.mean()

    def contrast(ours, other, target, e):
        a, b = comp_err(ours, target, e), comp_err(other, target, e)
        d = (b - a)
        sb = strat_boot(d.to_dict(), groups, rng)
        v = d.values; simple = v[rng.integers(0, 12, (NB, 12))].mean(1)
        pairs = sorted({g["pair"] for g in groups.values() if g["group"] == "pair"})
        lopo = {q: float(d[[m for m in d.index if groups[m]["pair"] != q]].mean()) for q in pairs}
        sub = {k: float(d[[m for m in d.index if f(groups[m])]].mean()) for k, f in
               (("pair_absorbers", lambda g: g["role"] == "absorber"), ("pair_transmitters", lambda g: g["role"] == "transmitter"),
                ("mixed", lambda g: g["group"] == "mixed"), ("Ho_rich", lambda g: g["group"] == "Ho-rich"))}
        return dict(target=target, energy_MeV=e, ours=ours, comparator=other, ours_MAPE=float(a.mean()), comparator_MAPE=float(b.mean()),
                    improvement=float(d.mean()), p5=float(np.quantile(sb, .05)), lo=float(np.quantile(sb, .025)), hi=float(np.quantile(sb, .975)),
                    p5_simple_bootstrap=float(np.quantile(simple, .05)), leave_one_pair_out_min=min(lopo.values()), **{"without_" + q: x for q, x in lopo.items()}, **sub)

    prim = pd.DataFrame([contrast("MLP-DR", c, t, 0.1) for t in ("EBF", "EABF") for c in ("MLP-Y2", "MLP-K1r")])
    prim["exceeds_margin"] = (prim.p5 > MARGIN) & (prim.p5_simple_bootstrap > MARGIN)  # both bootstraps, REGISTRATION.md section 6
    supported = bool(prim.exceeds_margin.all())
    sec = pd.DataFrame([contrast(o, c, t, e) for t in ("EBF", "EABF") for e in ENERGIES
                        for o, cs in (("MLP-DR", ("MLP-Y2", "MLP-K1r", "MLP-PAy", TREE[t])), ("MLP-PAy", ("MLP-Y2", "MLP-K1r")), ("tXGB-P1", ("tXGB-Y2", "tXGB-K1r")))
                        for c in cs if not (e == 0.1 and o == "MLP-DR" and c in ("MLP-Y2", "MLP-K1r"))])
    sec.to_csv(OUT / "contrasts.csv", index=False)

    gate, depthb = [], []
    for (m, t, e), d in j.groupby(["model", "target", "energy_MeV"]):
        b = d.groupby("material_id").res.mean()
        bse = d.assign(v=(d.se_pct * d.prediction / d.label) ** 2).groupby("material_id").v.apply(lambda v: np.sqrt(v.sum()) / len(v))
        w = d.loc[d.ape.idxmax()]; k = b.abs().idxmax()
        gate.append(dict(model=m, target=t, energy_MeV=e, MAPE=float(d.groupby("material_id").ape.mean().mean()), worst_bias=float(b.abs().max()), limiting_material=k,
                         signed=float(b[k]), signed_mc_se=float(bse[k]), n_over_5=int((b.abs() > GATE).sum()), passes=bool(b.abs().max() <= GATE),
                         worst_point=float(w.ape), worst_point_material=w.material_id, worst_point_depth=float(w.depth_mfp), worst_point_se=float(w.se_pct),
                         rms_z=float(np.sqrt((d.z ** 2).mean())), coverage=float(d.covered.mean()) if d.covered.notna().all() else np.nan))
        for x, dd in d.groupby("depth_mfp"):
            depthb.append(dict(model=m, target=t, energy_MeV=e, depth_mfp=x, MAPE=float(dd.ape.mean()), mean_bias=float(dd.res.mean()),
                               mean_bias_mc_se=float(np.sqrt(((dd.se_pct * dd.prediction / dd.label) ** 2).sum()) / len(dd)),
                               median_label_se=float(dd.se_pct.median()), max_label_se=float(dd.se_pct.max()),
                               noise_only_MAPE=float((dd.se_pct * np.sqrt(2 / np.pi)).mean()), rms_z=float(np.sqrt((dd.z ** 2).mean())),
                               coverage=float(dd.covered.mean()) if dd.covered.notna().all() else np.nan))
    gate, depthb = pd.DataFrame(gate), pd.DataFrame(depthb)
    gate.to_csv(OUT / "gate.csv", index=False); depthb.to_csv(OUT / "by_depth.csv", index=False)
    sub = j.groupby(["model", "target", "energy_MeV", "role"]).agg(MAPE=("ape", "mean"), mean_bias=("res", "mean")).reset_index()
    sub.to_csv(OUT / "by_role.csv", index=False)

    pr = []
    L = lab.set_index(KEYS)
    for q in sorted({g["pair"] for g in groups.values() if g["group"] == "pair"}):
        a_ = next(m for m, g in groups.items() if g["pair"] == q and g["role"] == "absorber")
        t_ = next(m for m, g in groups.items() if g["pair"] == q and g["role"] == "transmitter")
        for e in ENERGIES:
            for x in DEPTHS:
                for t in ("EBF", "EABF"):
                    ba, bt = L.loc[(a_, e, x)], L.loc[(t_, e, x)]
                    d = np.log(bt[t] / ba[t]); se = np.hypot(bt[t + "_se"] / bt[t], ba[t + "_se"] / ba[t])
                    pr.append(dict(pair=q, absorber=a_, transmitter=t_, energy_MeV=e, depth_mfp=x, target=t, difference_pct=100 * (np.exp(d) - 1), z=d / se))
    pr = pd.DataFrame(pr); pr.to_csv(OUT / "pair_contrasts.csv", index=False)

    out = dict(primary=dict(contrasts=prim.to_dict("records"), supported=supported), margin_points=MARGIN, gate_percent=GATE,
               bootstrap=dict(resamples=NB, seed=BOOT_SEED), registration_sha256=digest(OUT / "registration.json"),
               labels_sha256=digest(OUT / "labels.csv"), seed_processes=len(plan),
               gate_MLP_DR=gate[gate.model == "MLP-DR"].to_dict("records"))
    (OUT / "evaluation.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    order = ["MLP-DR", "MLP-PAy", "MLP-Y2", "MLP-K1r", "XGBoost-F", "tXGB-P1", "tXGB-Y2", "tXGB-K1r", "Reference alone"]
    srt = lambda d: d.assign(_o=d.model.map(order.index)).sort_values(["target", "energy_MeV", "_o"]).drop(columns="_o")
    dr = lambda d: d[d.model == "MLP-DR"].drop(columns="model")
    tail = lab[["material_id", "energy_MeV", "depth_mfp", "seeds"]].assign(
        EBF_se_pct=100 * lab.EBF_se / lab.EBF, EABF_se_pct=100 * lab.EABF_se / lab.EABF, EBF_largest_seed_over_mean=lab.EBF_largest_seed_over_mean,
        EBF_half_over_full=lab.EBF_mean_first_half / lab.EBF, EABF_half_over_full=lab.EABF_mean_first_half / lab.EABF, transport_ratio=lab.transport_ratio)
    R = ["# Prospective K-edge test of the frozen networks: evaluation (run once)", "",
         "Registration: REGISTRATION.md, registration.json. Percent throughout; improvements in percentage points.", "",
         "## Primary: 0.1 MeV, all 12 compositions, four-input network (MLP-DR) against the published-style networks", "",
         f"p5 = one-sided 95 % lower bound from the stratified bootstrap, p5_simple_bootstrap from resampling the 12 compositions; both must exceed the margin of {MARGIN} points.", "",
         md(prim[["target", "comparator", "ours_MAPE", "comparator_MAPE", "improvement", "p5", "lo", "hi", "exceeds_margin", "p5_simple_bootstrap", "leave_one_pair_out_min"]]), "",
         f"**Registered primary outcome: {'SUPPORTED' if supported else 'NOT SUPPORTED'}**", "",
         "Improvement by subgroup (descriptive):", "", md(prim[["target", "comparator", "pair_absorbers", "pair_transmitters", "mixed", "Ho_rich"]]), "",
         "## Material-mean bias (limit 5 %), worst point, standardized residuals and interval coverage", "",
         md(srt(gate)[["target", "energy_MeV", "model", "MAPE", "worst_bias", "limiting_material", "signed", "signed_mc_se", "n_over_5", "passes", "worst_point", "worst_point_depth", "worst_point_se", "rms_z", "coverage"]]), "",
         "## Four-input network by depth", "", md(dr(depthb).sort_values(["target", "energy_MeV", "depth_mfp"])), "",
         "## Mean error by composition role, 0.1 MeV", "", md(srt(sub[np.isclose(sub.energy_MeV, 0.1)])), "",
         "## Secondary contrasts", "", md(sec[["target", "energy_MeV", "ours", "comparator", "ours_MAPE", "comparator_MAPE", "improvement", "lo", "hi"]]), "",
         "## Transmitter against absorber in the four matched pairs (Geant4 labels)", "", md(pr.sort_values(["target", "energy_MeV", "depth_mfp", "pair"])), "",
         "## Achieved precision and tail diagnostics of the labels", "", md(tail.sort_values(["energy_MeV", "depth_mfp", "material_id"])), ""]
    (OUT / "RESULTS.md").write_text("\n".join(R) + "\n")
    print("\n".join(R[:14]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["select", "features", "predict", "register", "verify", "run", "evaluate"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    globals()[a.action](a) if a.action in ("features", "run") else globals()[a.action]()


if __name__ == "__main__":
    main()
