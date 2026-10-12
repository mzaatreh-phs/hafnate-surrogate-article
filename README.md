# Data and code: Fluorescence-Informed Surrogate Prediction of Gamma-Ray Buildup in Rare-Earth Hafnates

Archive for the article "Fluorescence-Informed Surrogate Prediction of Gamma-Ray Buildup in Rare-Earth Hafnates", M. Al-Zaatreh et al., in preparation for Radiation Physics and Chemistry.
Contact: mohammad.alzaatreh@gmail.com. The manuscript is not part of this repository.

## Files (release v2)

| File | Contents |
|---|---|
| `hafnate_surrogate_code_v2.tar.gz` | `AI_Haphinate/`: analysis, surrogate, cascade-model and figure scripts, material manifest; common to both articles of the study |
| `hafnate_surrogate_validation_v2.tar.gz` | `AI_Haphinate/validation/`: protocols, registrations and amendments, evaluation outputs, labels, frozen models, predictions |
| `hafnate_surrogate_geant4_runs_v2.tar.gz` | `BF/runs/`: Geant4 run records with seed- and campaign-level results, configurations, response tables and logs |
| `INDEX.csv` | Every analysis and run directory of the study: in this archive, in the companion archive `hafnate-deep-window-article`, or left out with the reason |
| `SHA256SUMS` | Checksums of the three tarballs; each tarball also contains a per-file SHA-256 manifest |

The tarballs extract into one tree, `AI_Haphinate/` and `BF/runs/`:

```
tar xzf hafnate_surrogate_code_v2.tar.gz
tar xzf hafnate_surrogate_validation_v2.tar.gz
tar xzf hafnate_surrogate_geant4_runs_v2.tar.gz
sha256sum -c SHA256SUMS
```

Source revisions: AI_Haphinate 132cc2b0674200bdbbe182f254bd062062be9162; BF Geant4 application 617d960b43a77595479bd2bd9d73e13060b47229, not archived.

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

To be stated before publication.
