#!/usr/bin/env python3
"""Step 2 of validation/nn_tests_20261010/PROTOCOL.md: oracle G-P fits to the development Geant4 depth curves.

F1 fit on 1-20 mfp; F2 fit on 1-10, predict 11-20; F3 fit on the eight analog shallow labels, predict 5-20.
Development data only (six deep-profile compositions); no held-out file is read.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/nn_tests_20261010/step2_oracle"
LO = np.array([1.0001, 0.05, -1.0, 2.0, -1.0])   # b, c, a, Xk, d
HI = np.array([30.0, 6.0, 1.0, 60.0, 1.0])
T2 = np.tanh(-2.0)


def gp(x, p):
    b, c, a, xk, d = p
    K = c * x ** a + d * (np.tanh(x / xk - 2.0) - T2) / (1.0 - T2)
    K = np.maximum(K, 1e-6)
    near = np.abs(K - 1.0) < 1e-7
    geo = np.where(near, x, (np.power(K, x) - 1.0) / np.where(near, 1.0, K - 1.0))
    return 1.0 + (b - 1.0) * geo


def fit(x, B, se, rng, starts=20):
    w = B / se
    res = lambda p: (np.log(np.maximum(gp(x, p), 1e-12)) - np.log(B)) * w
    best = None
    for k in range(starts):
        p0 = np.array([max(1.01, B[np.argmin(np.abs(x - 1))]), 1.0, 0.0, 14.0, 0.0]) if k == 0 else rng.uniform(LO, HI)
        try:
            r = least_squares(res, np.clip(p0, LO, HI), bounds=(LO, HI), max_nfev=400)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def main():
    rng = np.random.default_rng(20261010)
    deep = pd.read_csv(ROOT / "validation/deep_profile_20261007/step3/deep_labels.csv")
    sh = pd.read_csv(ROOT / "validation/highE_extension_20261003/surrogate_17E/training_data.csv")
    rows, pts = [], []
    for (m, e, t), g in deep.groupby(["material_id", "energy_MeV", "target"]):
        g = g.sort_values("depth_mfp")
        x, B, se = g.depth_mfp.values, g.B.values, g.B_se.values
        assert len(x) == 20
        # F1
        p1 = fit(x, B, se, rng)
        r1 = 100 * (gp(x, p1) / B - 1)
        z1 = (np.log(gp(x, p1)) - np.log(B)) * B / se
        # F2
        lo = x <= 10
        p2 = fit(x[lo], B[lo], se[lo], rng)
        r2 = 100 * (gp(x[~lo], p2) / B[~lo] - 1)
        # F3: analog shallow labels of the same composition and energy
        s = sh[(sh.material_id == m) & np.isclose(sh.energy_MeV, e)].sort_values("depth_mfp")
        assert len(s) == 8
        p3 = fit(s.depth_mfp.values, s[t].values, s[t + "_se"].values, rng)
        d5 = x >= 5
        r3 = 100 * (gp(x[d5], p3) / B[d5] - 1)
        rows.append(dict(material_id=m, energy_MeV=e, target=t, **{f"F1_{n}": v for n, v in zip("b c a Xk d".split(), p1)},
                         F1_max_abs_pct=np.abs(r1).max(), F1_mape=np.abs(r1).mean(), F1_rms_z=np.sqrt(np.mean(z1 ** 2)),
                         F2_mape_11_20=np.abs(r2).mean(), F2_max_abs_pct=np.abs(r2).max(),
                         F3_mape_5_20=np.abs(r3).mean(), F3_max_abs_pct=np.abs(r3).max(),
                         mean_rel_se_pct=100 * np.mean(se / B)))
        pts += [dict(material_id=m, energy_MeV=e, target=t, depth_mfp=xi, B=Bi, F1_pct=a_) for xi, Bi, a_ in zip(x, B, r1)]
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "oracle_fits.csv", index=False)
    pd.DataFrame(pts).to_csv(OUT / "oracle_F1_residuals.csv", index=False)
    L = ["# Step 2: oracle G-P fits to the development depth curves (run once, 2026-10-10)", ""]
    for t in ("EBF", "EABF"):
        g = r[r.target == t]
        L += [f"## {t}, {len(g)} curves", "",
              f"- F1 full fit, 1-20 mfp: curves with max |residual| <= 5 %: {100 * (g.F1_max_abs_pct <= 5).mean():.1f} %; "
              f"median max |residual| {g.F1_max_abs_pct.median():.2f} %, largest {g.F1_max_abs_pct.max():.2f} %; "
              f"median RMS weighted residual {g.F1_rms_z.median():.2f}; mean MAPE {g.F1_mape.mean():.2f} %",
              f"- F2 fit 1-10, predict 11-20 mfp: MAPE {g.F2_mape_11_20.mean():.2f} %, median {g.F2_mape_11_20.median():.2f} %, "
              f"largest curve MAPE {g.F2_mape_11_20.max():.1f} %, largest point error {g.F2_max_abs_pct.max():.1f} %",
              f"- F3 fit shallow 0.5-4, predict 5-20 mfp: MAPE {g.F3_mape_5_20.mean():.1f} %, median {g.F3_mape_5_20.median():.1f} %, "
              f"largest curve MAPE {g.F3_mape_5_20.max():.0f} %",
              f"- mean relative SE of the profile labels {g.mean_rel_se_pct.mean():.2f} %", ""]
        L += ["F2 MAPE by energy (%): " + ", ".join(f"{e:g} MeV {v:.1f}" for e, v in g.groupby("energy_MeV").F2_mape_11_20.mean().items()), ""]
    ok1 = all((r[r.target == t].F1_max_abs_pct <= 5).mean() >= 0.9 and r[r.target == t].F1_rms_z.median() <= 1.5 for t in ("EBF", "EABF"))
    ok2 = all(r[r.target == t].F2_mape_11_20.mean() <= 5 for t in ("EBF", "EABF"))
    L += ["## Decision under the protocol", "",
          f"- Adequate as a fitted representation: **{'YES' if ok1 else 'NO'}**",
          f"- Adequate for depth extrapolation, 1-10 to 11-20 mfp: **{'YES' if ok2 else 'NO'}**"]
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
