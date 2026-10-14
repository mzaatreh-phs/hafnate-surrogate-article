#!/usr/bin/env python3
"""Isolated prospective validation: prepare and freeze, run BF, then evaluate.

Preparation reads development labels only. No fresh simulation labels are read
until evaluate. BF source, production materials and development models remain
unchanged. The run action must be explicitly invoked after reviewing registration.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BF = ROOT.parent / 'BF'
DEV = ROOT / 'validation/physics_surrogate_full_range_20260930'
OUT = DEV / 'fresh_confirmation'
PLAN = ROOT / 'validation/physics_surrogate_20260930/fresh_validation_plan.json'
sys.path.insert(0, str(ROOT / 'ml'))
sys.path.insert(0, str(ROOT / 'scripts'))
import fluorescence_features as FF
import response_tables as RT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(out):
    reg = json.loads((out / 'PREREGISTRATION.json').read_text())
    if digest(out / 'PREREGISTRATION.json') != (out / 'registration.sha256').read_text().strip():
        raise RuntimeError('Registration changed')
    for name, sha in reg['frozen_sha256'].items():
        if digest(out / name) != sha:
            raise RuntimeError(f'Frozen file changed: {name}')
    # The production atomic data are external; verify their registered content.
    for name, sha in reg['geant4_atomic_data_sha256'].items():
        if digest(Path(reg['G4LEDATA']) / name) != sha:
            raise RuntimeError(f'Geant4 atomic data changed: {name}')
    return reg


def anchor_task(task):
    mid, weights, tables, air, depths = task
    FF.F.TABLES = Path(tables)
    FF.F.AIR = Path(air)
    return FF.material_features(mid, weights, energies=[.1], depths=depths)


def prepare(out, workers):
    if (out / 'PREREGISTRATION.json').exists():
        verify(out)
        print('Existing registration verified; preparation is unchanged.', flush=True)
        return
    if (out / 'runs').exists():
        raise RuntimeError('Fresh run directory exists before registration')
    start = time.perf_counter()
    source = DEV / 'frozen_for_confirmation'
    metadata = json.loads((source / 'metadata.json').read_text())
    for name, sha in metadata['sha256'].items():
        path = (ROOT / 'scripts/physics_surrogate_full_range.py' if name == 'benchmark_script'
                else DEV / 'freeze_for_confirmation.py' if name == 'selection_script'
                else source / name)
        if digest(path) != sha:
            raise RuntimeError(f'Development artifact changed: {path}')
    recipes = json.loads(PLAN.read_text())['materials']
    ids = [m['material_id'] for m in recipes]
    assert len(ids) == 6 and len(set(ids)) == 6
    assert not set(ids) & set(metadata['training_materials'])
    energies = metadata['energies_MeV']; depths = metadata['depths_MFP']
    assert len(energies) == 13 and energies[0] == .1 and energies[-1] == 5 and len(depths) == 8
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / 'frozen'; frozen.mkdir(exist_ok=True)
    models = frozen / 'models'; models.mkdir(exist_ok=True)
    for path in source.iterdir():
        if path.is_file(): shutil.copy2(path, models / path.name)
    shutil.copy2(source / 'training_data.csv', frozen / 'development_data.csv')
    shutil.copy2(PLAN, frozen / 'original_recipe_plan.json')
    shutil.copy2(Path(__file__), frozen / 'confirm_physics_full_range.py')
    shutil.copy2(ROOT / 'scripts/physics_surrogate_full_range.py', frozen / 'benchmark.py')
    shutil.copy2(DEV / 'freeze_for_confirmation.py', frozen / 'selection.py')
    shutil.copy2(BF / 'scripts/generate_labels.py', frozen / 'bf_driver.py')
    binary = BF / 'build_deep_validation/bin/BF'
    shutil.copy2(binary, frozen / 'BF_snapshot')
    for path in [ROOT / 'ml/fluorescence_features.py', ROOT / 'scripts/fluo_model.py',
                 ROOT / 'scripts/radial_quadrature.py', ROOT / 'scripts/response_tables.py',
                 ROOT / 'study_policy.json']:
        shutil.copy2(path, frozen / path.name)
    save_json(frozen / 'recipes.json', recipes)
    rows = []
    for m in recipes:
        w = m['element_mass_fractions']
        assert abs(sum(w.values()) - 1) < 1e-10
        rows.append(dict(material_id=m['material_id'], density_g_cm3=m['density_g_cm3'],
                         **{e+'_wt_fraction':w[e] for e in FF.F.ELEMENTS}))
    pd.DataFrame(rows).to_csv(frozen / 'materials.csv', index=False)
    # Build edge-preserving approximate compound response tables from archived
    # NIST elemental inputs, without modifying any production table directory.
    raw_source = ROOT / 'validation/response_20260928/raw'
    if not raw_source.exists(): raw_source = ROOT / 'validation/response_20260927/raw'
    raw = frozen / 'nist_raw'; raw.mkdir(exist_ok=True)
    elements = {e for m in recipes for e,w in m['element_mass_fractions'].items() if w}
    tables = {}
    for el in sorted(elements):
        name = f'z{RT.Z[el]:02d}.html'
        shutil.copy2(raw_source / name, raw / name)
        tables[el] = RT.parse_table((raw / name).read_text())
    response = frozen / 'response_tables'; response.mkdir(exist_ok=True)
    air_source = ROOT / 'validation/response_20260928/reference_air.csv'
    if not air_source.exists(): air_source = ROOT / 'validation/response_20260927/reference_air.csv'
    shutil.copy2(air_source, frozen / 'reference_air.csv')
    for m in recipes:
        weights = {e:w for e,w in m['element_mass_fractions'].items() if w}
        knots = sorted({r[0] for e in weights for r in tables[e]} | {.6617,1.25,2.})
        edges = {x for e in weights for x,n in Counter(r[0] for r in tables[e]).items() if n == 2}
        grid = sorted(set(knots) | {a*(b/a)**(j/8) for a,b in zip(knots,knots[1:]) for j in range(1,8)})
        values = []
        for energy in grid:
            if energy in edges: values.append((energy,RT.mixture(tables,weights,energy,'left')))
            values.append((energy,RT.mixture(tables,weights,energy)))
        RT.write_csv(response / (m['material_id']+'.csv'), ['Energy_MeV','mu_en_rho_cm2_g'], values)
    tasks = [(m['material_id'],m['element_mass_fractions'],str(response),str(frozen/'reference_air.csv'),depths) for m in recipes]
    feature_rows = []; diagnostics = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for m,(values,diag) in zip(recipes,pool.map(anchor_task,tasks)):
            feature_rows.extend(values); diagnostics.append(diag)
            print('Prepared independent 100-keV anchor',m['material_id'],flush=True)
    physical = pd.DataFrame(feature_rows)
    physical.to_csv(frozen / 'physical_features.csv', index=False)
    save_json(frozen / 'anchor_diagnostics.json', diagnostics)
    anchor = physical.drop(columns='energy_MeV').rename(columns={'physics_EBF':'anchor_EBF','physics_EABF':'anchor_EABF'})
    coordinates = pd.DataFrame([dict(material_id=m['material_id'],energy_MeV=e,depth_mfp=d,
        density_g_cm3=m['density_g_cm3'],log10_E_MeV=np.log10(e),
        **{el+'_wt':m['element_mass_fractions'][el] for el in FF.F.ELEMENTS}) for m in recipes for e in energies for d in depths])
    frame = coordinates.merge(anchor,on=['material_id','depth_mfp'],validate='many_to_one')
    frame.to_csv(frozen / 'prediction_inputs.csv',index=False)
    training = pd.read_csv(frozen / 'development_data.csv')
    comp = ['density_g_cm3'] + [e+'_wt' for e in FF.F.ELEMENTS]
    material = training.drop_duplicates('material_id').set_index('material_id')[comp]
    scaler = StandardScaler().fit(material)
    joblib.dump(scaler,frozen / 'neighbor_scaler.joblib')
    predictions = []; neighbors = {}
    prediction_start = time.perf_counter()
    bands = pd.read_csv(models / 'empirical_bands.csv').set_index(['target','energy_MeV'])
    for mid in ids:
        test = frame[frame.material_id == mid]
        distance = np.linalg.norm(scaler.transform(material)-scaler.transform(test[comp].iloc[:1]),axis=1)
        idx = np.argsort(distance,kind='stable')[:3]; nearest = material.index[idx]
        weight = 1/np.maximum(distance[idx],1e-12); weight /= sum(weight)
        neighbors[mid] = dict(materials=nearest.tolist(),distances=distance[idx].tolist(),weights=weight.tolist())
        index = pd.MultiIndex.from_frame(test[['energy_MeV','depth_mfp']])
        for target in ['EBF','EABF']:
            ys = [training[training.material_id == n].set_index(['energy_MeV','depth_mfp']).loc[index,target].to_numpy() for n in nearest]
            methods = {'3NN':np.exp(weight @ np.log(ys))}
            for method in dict.fromkeys(['XGB','anchor_residual','full_residual',metadata['selected_methods'][target]]):
                name = f'{target}_{method}.joblib'; definition = metadata['model_definitions'][name]
                model = joblib.load(models / name)
                x = test.copy(); x['log_anchor'] = np.log(x['anchor_'+target])
                logp = model.predict(x[definition['features']]).astype(float)
                if definition['residual']: logp += x.log_anchor.to_numpy()
                methods[method] = np.maximum(1.,np.exp(logp))
            for method,values in methods.items():
                for (_,r),p in zip(test.iterrows(),values):
                    record=dict(material_id=mid,target=target,method=method,
                        energy_MeV=float(r.energy_MeV),depth_mfp=float(r.depth_mfp),prediction=float(p))
                    if method==metadata['selected_methods'][target]:
                        q=float(bands.loc[(target,float(r.energy_MeV)),'absolute_log_error_quantile_90'])
                        record.update(lower90=max(1.,float(p)*np.exp(-q)),upper90=float(p)*np.exp(q))
                    predictions.append(record)
    prediction_seconds = time.perf_counter()-prediction_start
    pd.DataFrame(predictions).to_csv(frozen / 'predictions.csv',index=False)
    save_json(frozen / 'neighbors.json',neighbors)
    driver = load_module(frozen / 'bf_driver.py','fresh_bf_driver_prepare')
    macros = frozen / 'macros'; macros.mkdir(exist_ok=True)
    seeds = []
    for row in rows:
        for energy in energies:
            for seed in range(32):
                pair = driver.seed_pair(row['material_id'],f'{energy:g}',seed)
                name = f"{row['material_id']}_E{energy:g}_seed{seed:03d}.mac"
                (macros / name).write_text(driver.material_macro(row,f'{energy:g}',[f'{d:g}' for d in depths],10000,pair,.01,None))
                seeds.append(dict(material_id=row['material_id'],energy_MeV=energy,index=seed,seeds=pair,macro=name))
    save_json(frozen / 'seeds.json',seeds)
    g4data = ROOT.parent / 'geant4/install/share/Geant4/data/G4EMLOW8.8'
    if not g4data.exists(): raise FileNotFoundError(g4data)
    # Full installed EM data content is registered, including photoelectric,
    # conversion, Compton and deexcitation data. No expensive simulations here.
    atomic_hashes = {str(p.relative_to(g4data)):digest(p) for p in sorted(g4data.rglob('*')) if p.is_file()}
    registration = dict(status='prospective preregistration; fresh labels not generated',
        registered_utc=datetime.now(timezone.utc).isoformat(),selected_methods=metadata['selected_methods'],
        materials=ids,energies_MeV=energies,depths_MFP=depths,
        material_scope='six hypothetical homogeneous hafnate compositions; density approximated by reciprocal mixture; synthesis and family extrapolation untested',
        reference_anchor_energy_MeV=.1,anchor_scope='reference-energy normalization/features; not query-energy transport; matched MFP depths imply different physical radii',
        fixed_seeds_per_material_energy=32,events_per_seed_per_depth=10000,
        production_histories=6*13*32*8*10000,source_material_energy_configurations=78,seed_processes=2496,
        physics='Geant4 11.4.2 G4EmStandardPhysics_option4; analog, fluorescence/Auger/PIXE true; production cut 0.01 mm; thin shell fraction default 0.0002; point isotropic sphere',
        response_policy='NIST additive elemental mu_en approximation; duplicate edges preserved; lower boundary; no transport labels used in anchor',
        G4LEDATA=str(g4data.resolve()),geant4_atomic_data_sha256=atomic_hashes,
        primary_domain='all 13 energies, eight depths, equal-weight material-macro MAPE',
        primary_criteria=dict(full_grid_MAPE_ratio_max=1.05,near_edge_relative_MAPE_reduction_min=.20,
                              comparators=['XGB','3NN'],both_targets_required=True,
                              worst_absolute_material_bias_at_0_1_MeV_percent_max=5.),
        protocol_revision='Before fresh labels: development candidate selection constrained by worst absolute 0.1-MeV bias <=5%; full-grid noninferiority and near-edge improvement replace the preliminary global 20% reduction plan. Original unconstrained development selection remains archived.',
        secondary_domains=['0.1 MeV','each sampled energy','anchor_residual and full_residual ablations'],
        interval_calibration='fixed empirical 90% development log-error bands for selected methods only; not conformal or calibrated population intervals',
        descriptive_coverage_flag_min_percent=85.,coverage_domains=['all_13E','0.1_MeV'],
        uncertainty='independent-seed SE; paired material bootstrap and nested seed bootstrap preserving depth correlations within each material-energy process; descriptive, six materials only',
        bootstrap_replicates=5000,bootstrap_rng_seed=20260930,
        prohibitions=['no retraining or model selection after fresh labels','no dropping seeds or flagged points','no seed extensions for significance','no replacement of failed primary criterion with secondary success'],
        runtime=dict(preparation_seconds=time.perf_counter()-start,anchor_seconds=[d['wall_s'] for d in diagnostics],
                     prediction_batch_seconds=prediction_seconds,interpretation='includes model loading and DataFrame construction; not pure inference latency'),
        frozen_sha256={str(p.relative_to(out)):digest(p) for p in sorted(frozen.rglob('*')) if p.is_file()})
    save_json(out / 'PREREGISTRATION.json',registration)
    (out / 'registration.sha256').write_text(digest(out / 'PREREGISTRATION.json')+'\n')
    (out / 'PREREGISTRATION.md').write_text('# Prospective full-range surrogate confirmation\n\n'
        'Predictions, selected methods, controls, recipes, response tables, source snapshots, executable, atomic data and all 2,496 seed macros are frozen before fresh labels.\n\n'
        f"Selected methods: {metadata['selected_methods']}. All 13 sampled energies from 0.1 to 5 MeV and eight depths from 0.5 to 4 MFP are primary.\n\n"
        'Success requires full-grid material-macro MAPE at most 1.05 times each XGB and 3NN comparator, at least 20% relative MAPE reduction at 0.1 MeV against both comparators, and worst absolute material bias at 0.1 MeV at most 5%, for both targets. No exclusions or seed extensions.\n\n'
        'Before any fresh labels, development selection was constrained to candidates with worst 0.1-MeV material bias at most 5%. Full-grid noninferiority and near-edge improvement replace the preliminary global 20% reduction criterion. The unconstrained development selection is retained separately.\n\n'
        'Budget: 78 material-energy configurations × 32 seeds × eight independent depth BeamOn runs × 10,000 histories = 199,680,000 production histories, plus startup warmups.\n\n'
        'These are hypothetical computational compositions and approximate densities. The 100-keV atomic anchor supplies reference features, not full-range transport. NIST compound responses use elemental additivity. Bootstrap intervals are descriptive with only six materials.\n\n'
        'The JSON registration and SHA-256 record define the complete protocol. Explicitly invoke run after reviewing them.\n')
    verify(out)
    print(f'Prepared and verified {out}; no fresh BF labels launched.',flush=True)


def run(out,jobs):
    reg = verify(out); frozen = out/'frozen'
    driver = load_module(frozen/'bf_driver.py','fresh_bf_driver_run')
    driver.ROOT = BF; driver.MATERIALS = frozen/'materials.csv'
    sys.argv = ['bf_driver.py','--output',str(out/'runs'),'--materials',','.join(reg['materials']),
        '--energies',','.join(f'{e:g}' for e in reg['energies_MeV']),
        '--mfps',','.join(f'{d:g}' for d in reg['depths_MFP']), '--events','10000',
        '--min-seeds','32','--max-seeds','32','--jobs',str(jobs),'--cut-mm','.01',
        '--table-dir',str(frozen/'response_tables'),'--binary',str(frozen/'BF_snapshot')]
    os.environ['G4LEDATA'] = reg['G4LEDATA']
    original_run_seed = driver.run_seed
    def timed_run_seed(args,binary,env,row,energy,index):
        folder=args.output/row['material_id']/f'E{energy}'/f'seed_{index:03d}'
        resumed=(folder/'complete.json').exists()
        begin=time.perf_counter()
        result=original_run_seed(args,binary,env,row,energy,index)
        macro=out/'frozen/macros'/f"{row['material_id']}_E{energy}_seed{index:03d}.mac"
        if digest(folder/'m.txt') != digest(macro):
            raise RuntimeError(f'Run macro differs from registration: {folder}')
        if not resumed:
            save_json(folder/'runtime.json',dict(wall_seconds=time.perf_counter()-begin,
                completed_utc=datetime.now(timezone.utc).isoformat()))
        return result
    driver.run_seed=timed_run_seed
    start = time.time()
    save_json(out/'run_started.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,
        registration_sha256=digest(out/'PREREGISTRATION.json'),command=sys.argv))
    driver.main()
    verify(out)
    save_json(out/'run_completed.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.time()-start,
        labels_sha256=digest(out/'runs/labels.csv')))


def evaluate(out):
    reg = verify(out); labels = pd.read_csv(out/'runs/labels.csv')
    keys = ['material_id','energy_MeV','depth_mfp']
    expected = pd.read_csv(out/'frozen/prediction_inputs.csv')[keys]
    assert len(labels)==624 and not labels.duplicated(keys).any() and (labels.seeds==32).all()
    assert len(expected.merge(labels,on=keys,validate='one_to_one'))==624
    prediction = pd.read_csv(out/'frozen/predictions.csv')
    errors = []; metrics = []; material_metrics = []
    for target in ['EBF','EABF']:
        rows = prediction[prediction.target==target].merge(labels[keys+[target,target+'_se','flags']],on=keys,validate='many_to_one')
        rows['relative_error'] = rows.prediction/rows[target]-1
        errors.append(rows)
        for method,g in rows.groupby('method'):
            for domain,h in [('all_13E',g)]+[(f'{e:g}_MeV',g[g.energy_MeV==e]) for e in reg['energies_MeV']]:
                scores = h.groupby('material_id').relative_error.agg(mape=lambda x:100*abs(x).mean(),bias=lambda x:100*x.mean())
                metrics.append(dict(target=target,method=method,domain=domain,MAPE_percent=scores.mape.mean(),
                    bias_percent=scores.bias.mean(),worst_abs_material_bias_percent=abs(scores.bias).max()))
                material_metrics.extend(dict(target=target,method=method,domain=domain,material_id=mid,**r.to_dict()) for mid,r in scores.iterrows())
    pd.concat(errors).to_csv(out/'prediction_errors.csv',index=False)
    scores=pd.DataFrame(metrics);scores.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(material_metrics).to_csv(out/'material_metrics.csv',index=False)
    coverage=[]
    for target in ['EBF','EABF']:
        chosen=pd.concat(errors)
        chosen=chosen[(chosen.target==target)&(chosen.method==reg['selected_methods'][target])].copy()
        chosen['covered']=(chosen[target]>=chosen.lower90)&(chosen[target]<=chosen.upper90)
        for domain,h in [('all_13E',chosen),('0.1_MeV',chosen[chosen.energy_MeV==.1])]:
            percent=100*float(h.covered.mean())
            coverage.append(dict(target=target,domain=domain,coverage_percent=percent,
                below_descriptive_85_percent_flag=percent<85,points=len(h)))
    pd.DataFrame(coverage).to_csv(out/'empirical_band_coverage.csv',index=False)
    # Read and verify every seed record. A shared resample across all depths
    # preserves the depth correlations produced by one seeded BF process.
    raw=[];nboot=reg['bootstrap_replicates'];rng=np.random.default_rng(reg['bootstrap_rng_seed'])
    simulated={t:np.empty((nboot,len(expected))) for t in ['EBF','EABF']}
    for mid in reg['materials']:
        for energy in reg['energies_MeV']:
            per_seed=[]
            for i in range(32):
                folder=out/'runs'/mid/f'E{energy:g}'/f'seed_{i:03d}'
                marker=json.loads((folder/'complete.json').read_text())
                for name,sha in marker['files'].items(): assert digest(folder/name)==sha
                assert marker['seeds']==load_module(out/'frozen/bf_driver.py','seed_check').seed_pair(mid,f'{energy:g}',i)
                rows=marker['rows'];assert len(rows)==8
                per_seed.append(rows)
                raw.extend(dict(material_id=mid,energy_MeV=energy,seed_index=i,**r) for r in rows)
            indices=rng.integers(0,32,size=(nboot,32))
            columns=expected.index[(expected.material_id==mid)&(expected.energy_MeV==energy)].to_numpy()
            for target in simulated:
                values=np.array([[float(r[target+'_TL']) for r in seed] for seed in per_seed])
                assert np.allclose(values.mean(axis=0),labels.set_index(keys).loc[(mid,energy),target].to_numpy())
                simulated[target][:,columns]=values[indices].mean(axis=1)
    pd.DataFrame(raw).to_csv(out/'raw_seed_values.csv',index=False)
    material_resample=rng.integers(0,6,size=(nboot,6));comparisons=[]
    for target in simulated:
        for domain,mask in [('all_13E',np.ones(624,dtype=bool)),('0.1_MeV',(expected.energy_MeV==.1).to_numpy())]:
            boot_scores={};point_scores={}
            for method in prediction[prediction.target==target].method.unique():
                p=expected.merge(prediction[(prediction.target==target)&(prediction.method==method)],on=keys,validate='one_to_one').prediction.to_numpy()
                err=100*abs(p[None,:]/simulated[target]-1)
                mats=np.stack([err[:,mask&(expected.material_id==mid).to_numpy()].mean(axis=1) for mid in reg['materials']],axis=1)
                boot_scores[method]=mats[np.arange(nboot)[:,None],material_resample].mean(axis=1)
                point_scores[method]=float(scores[(scores.target==target)&(scores.method==method)&(scores.domain==domain)].iloc[0].MAPE_percent)
            selected=reg['selected_methods'][target]
            for comparator in ['XGB','3NN','anchor_residual','full_residual']:
                if comparator==selected:continue
                difference=boot_scores[selected]-boot_scores[comparator]
                reduction=1-boot_scores[selected]/boot_scores[comparator]
                comparisons.append(dict(target=target,domain=domain,selected=selected,comparator=comparator,
                    MAPE_difference_pp=point_scores[selected]-point_scores[comparator],
                    relative_MAPE_reduction=1-point_scores[selected]/point_scores[comparator],
                    difference_lower_pp=float(np.quantile(difference,.025)),difference_upper_pp=float(np.quantile(difference,.975)),
                    reduction_lower=float(np.quantile(reduction,.025)),reduction_upper=float(np.quantile(reduction,.975))))
    comparisons=pd.DataFrame(comparisons);comparisons.to_csv(out/'paired_bootstrap.csv',index=False)
    primary=[]
    for target in ['EBF','EABF']:
        p=comparisons[(comparisons.target==target)&(comparisons.domain=='all_13E')&comparisons.comparator.isin(['XGB','3NN'])]
        near=comparisons[(comparisons.target==target)&(comparisons.domain=='0.1_MeV')&comparisons.comparator.isin(['XGB','3NN'])]
        bias=float(scores[(scores.target==target)&(scores.method==reg['selected_methods'][target])&(scores.domain=='0.1_MeV')].iloc[0].worst_abs_material_bias_percent)
        primary.append(dict(target=target,full_grid_comparisons=p.to_dict('records'),near_edge_comparisons=near.to_dict('records'),worst_0_1_MeV_abs_bias_percent=bias,
            passes_registered_point_criteria=bool((1-p.relative_MAPE_reduction<=1.05).all() and (near.relative_MAPE_reduction>=.2).all() and bias<=5)))
    result=dict(primary=primary,overall_pass=all(p['passes_registered_point_criteria'] for p in primary),
        flagged_points=int(labels['flags'].fillna('').ne('').sum()),
        registration_sha256=digest(out/'PREREGISTRATION.json'),labels_sha256=digest(out/'runs/labels.csv'),
        uncertainty='nested independent-seed and paired-material bootstrap; only six materials; intervals descriptive')
    result['empirical_band_coverage']=coverage
    save_json(out/'evaluation.json',result)
    report=['# Fresh full-range surrogate confirmation','',f"Registered primary outcome: {'PASS' if result['overall_pass'] else 'FAIL'}.",'',
        '| Target | Selected method | Full-grid MAPE (%) | XGB MAPE (%) | 3NN MAPE (%) | Worst 0.1-MeV absolute bias (%) |',
        '|---|---|---:|---:|---:|---:|']
    for p in primary:
        target=p['target'];v=[float(scores[(scores.target==target)&(scores.method==m)&(scores.domain=='all_13E')].iloc[0].MAPE_percent) for m in [reg['selected_methods'][target],'XGB','3NN']]
        report.append(f"| {target} | {reg['selected_methods'][target]} | "+' | '.join(f'{x:.3f}' for x in v)+f" | {p['worst_0_1_MeV_abs_bias_percent']:.3f} |")
    report+=['',f"All 624 points and 32 seeds retained; {result['flagged_points']} points carry precision/estimator flags.",
        'Paired comparisons and nested seed/material bootstrap intervals are in paired_bootstrap.csv. Six compositions limit population claims.',
        'The targets use hypothetical material definitions and approximate additive NIST responses. Continuous energy, synthesis, new chemical families and deep extrapolation remain unvalidated.']
    (out/'RESULTS.md').write_text('\n'.join(report)+'\n');print('\n'.join(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','verify','run','evaluate'])
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--jobs',type=int,default=16)
    args=parser.parse_args();out=args.output.resolve()
    if args.action=='prepare':prepare(out,args.workers)
    elif args.action=='verify':verify(out);print('Frozen registration verified.')
    elif args.action=='run':run(out,args.jobs)
    else:evaluate(out)


if __name__=='__main__':main()
