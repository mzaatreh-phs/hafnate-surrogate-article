# Data and code: Fluorescence-Informed Surrogate Prediction of Gamma-Ray Buildup in Rare-Earth Hafnates

Archive for the article "Fluorescence-Informed Surrogate Prediction of Gamma-Ray Buildup in Rare-Earth Hafnates", M. Al-Zaatreh et al., in preparation for Radiation Physics and Chemistry.
Contact: mohammad.alzaatreh@gmail.com. The manuscript is not part of this repository.

## Files (release v5)

| File | Contents |
|---|---|
| `hafnate_surrogate_code_v5.tar.gz` | `AI_Haphinate/`: analysis, surrogate, cascade-model and figure scripts, material manifest; common to both articles of the study |
| `hafnate_surrogate_validation_v5.tar.gz` | `AI_Haphinate/validation/`: protocols, registrations and amendments, evaluation outputs, labels, frozen models, predictions |
| `hafnate_surrogate_geant4_runs_v5.tar.gz` | `BF/runs/`: Geant4 run records with seed- and campaign-level results, configurations, response tables and logs |
| `INDEX.csv` | Every analysis and run directory of the study: in this archive, in the companion archive `hafnate-deep-window-article`, or left out with the reason |
| `SHA256SUMS` | Checksums of the three tarballs; each tarball also contains a per-file SHA-256 manifest |

The tarballs extract into one tree, `AI_Haphinate/` and `BF/runs/`:

```
tar xzf hafnate_surrogate_code_v5.tar.gz
tar xzf hafnate_surrogate_validation_v5.tar.gz
tar xzf hafnate_surrogate_geant4_runs_v5.tar.gz
sha256sum -c SHA256SUMS
```

Source revisions: AI_Haphinate 8af4af657292dbea21e1ab103ea069e701a54cce; BF Geant4 application 617d960b43a77595479bd2bd9d73e13060b47229, not archived.

## Browsable files in this repository

Besides the release tarballs, the following files can be read directly here. They are extracted from the same
sources as release v5 by `AI_Haphinate/scripts/export_repo_files.py`.

| Path | Contents |
|---|---|
| `geant4/seed_results/<run>.csv.gz` | Every seed-level Geant4 result of the article's seed runs, one file per run: track-length EBF and EABF (`EBF_TL`, `EABF_TL`), crossing estimators (`EBF`, `EABF_mat`), uncollided check, relative errors, attenuation coefficient, wall time, with the material, energy folder, seed and run variant. Each seed is 10 000 histories unless `Events` says otherwise. The label of a point is the mean over its seeds and its standard error the standard error of that mean; for the development labels this reproduces `geant4/tables/development_training_data.csv` exactly. |
| `geant4/seed_results/fresh_test_20260930.csv.gz` | Seed values of the fresh-composition test |
| `geant4/tables/` | Label tables used by the network: development training data (40 compositions, 17 energies, 8 depths), held-out labels with predictions, prospective labels, compositions and predictions, the K-edge replication, the La/Gd benchmark against ANS-6.4.3 and the Bearden line-energy check |
| `network/code/AI_Haphinate/` | The network scripts (`scripts/nn_*.py`), the prospective-test script, the fluorescence-reference code (`scripts/fluo_model.py`, `scripts/radial_quadrature.py`, `ml/fluorescence_features.py`), the response-table code, the material manifest and the study policy |
| `network/frozen_models/heldout_evaluation/` | Frozen networks (`*.pt`, PyTorch) and tuned XGBoost models (`*.json`) of the held-out evaluation, with their settings (`*_selection.json`), predictions and SHA-256 manifest; `MLP-DR` is the four-input network |
| `network/frozen_models/prospective_test/` | Inputs, predictions, seed plan and hashes saved before the prospective transport runs |

The Geant4 run folders with macros, logs, configurations and spectra, and the staged deep-transport segments, are
in the release tarballs only.

## Scope

- Included: the work used by this article, carried out after the BF change that removes coherent scattering from
  the transport and from the mean-free-path attenuation coefficient, the ANS-6.4.3 convention. Directories used by
  both articles of the study are in both archives.
- Left out: runs with the superseded convention, internal review logs, runs of unrelated materials and work used
  by neither article; `INDEX.csv` lists each with the reason.
- Photon banks of the staged deep-transport runs and binary snapshots of the BF executable are not archived because
  of their size. The campaign results built from them are included.
- The source code of the BF Geant4 application, including its driver scripts, Geant4 patches and modified G4EMLOW
  files, is not part of this archive; copies of BF driver scripts inside run and registration folders are removed.
  The run records keep the configurations, macros, seeds, logs and results. The BF source is available from the
  author on reasonable request.

## Licence

MIT License, see `LICENSE`. It covers the code, data, models and archives of this repository, including the files of the release tarballs.
