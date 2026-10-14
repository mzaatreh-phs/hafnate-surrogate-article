#!/usr/bin/env python3
"""Archive NIST tables and build explicitly approximate, edge-preserving mixtures."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
Z={'La':57,'Ce':58,'Pr':59,'Nd':60,'Sm':62,'Eu':63,'Gd':64,'Tb':65,'Dy':66,'Ho':67,'Er':68,'Tm':69,'Y':39,'Yb':70,'Lu':71,'Sc':21,'Hf':72,'O':8,'H':1,'C':6,'N':7,'Ar':18,'S':16,'Si':14,'Ti':22,'As':33,'Pb':82}
BASE='https://pml.nist.gov/PhysRefData/XrayMassCoef/'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def parse_table(text):
    block=re.search(r'<pre[^>]*>(.*?)</pre>',text,re.I|re.S)
    if not block:raise ValueError('Missing NIST ASCII table')
    plain=html.unescape(re.sub(r'<[^>]+>','',block[1]))
    rows=[]
    for line in plain.splitlines():
        numbers=re.findall(r'\d+\.\d+[Ee][+-]\d+',line)
        if len(numbers)==3:rows.append(tuple(map(float,numbers)))
    if len(rows)<2:raise ValueError('Too few NIST data rows')
    if any(not all(math.isfinite(x) and x>0 for x in r) for r in rows):raise ValueError('Invalid values')
    if any(a[0]>b[0] for a,b in zip(rows,rows[1:])):raise ValueError('Unsorted table')
    if max(Counter(r[0] for r in rows).values())>2:raise ValueError('Ambiguous duplicate energy')
    return rows

def interpolate(rows,energy,column=2,side='right'):
    if side not in ('left','right'):raise ValueError('Invalid edge side')
    energies=[r[0] for r in rows]
    if not math.isfinite(energy) or energy<energies[0] or energy>energies[-1]:raise ValueError('Energy outside table')
    left,right=bisect_left(energies,energy),bisect_right(energies,energy)
    if left!=right:return rows[left if side=='left' else right-1][column]
    lo,hi=rows[left-1],rows[left]
    t=math.log(energy/lo[0])/math.log(hi[0]/lo[0])
    return math.exp(math.log(lo[column])+t*math.log(hi[column]/lo[column]))

def mixture(tables,weights,energy,side='right'):
    if not weights or any(not math.isfinite(w) or w<0 for w in weights.values()) or abs(sum(weights.values())-1)>1e-8:
        raise ValueError('Invalid mass fractions')
    return math.fsum(w*interpolate(tables[el],energy,side=side) for el,w in weights.items() if w)

def write_csv(path,fields,rows):
    with path.open('w',newline='') as out:
        writer=csv.writer(out);writer.writerow(fields);writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'validation/response_20260908')
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args();out=args.output;raw=out/'raw';raw.mkdir(parents=True,exist_ok=True)
    targets={f'z{z:02d}':f'ElemTab/z{z:02d}.html' for z in Z.values()}
    targets.update(air='ComTab/air.html',water='ComTab/water.html',gadolinium='ComTab/gadolinium.html',glass='ComTab/glass.html')
    def fetch(item):
        name,relative=item;path=raw/f'{name}.html';url=BASE+relative
        if not path.exists():
            if args.offline:raise FileNotFoundError(path)
            data=urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=45).read()
            parse_table(data.decode())
            path.write_bytes(data)
        rows=parse_table(path.read_text())
        return name,rows,dict(url=url,path=str(path.resolve()),sha256=digest(path))
    with ThreadPoolExecutor(max_workers=4) as pool:loaded=list(pool.map(fetch,targets.items()))
    parsed={name:rows for name,rows,_ in loaded};tables={el:parsed[f'z{z:02d}'] for el,z in Z.items()}
    candidates=out/'candidate_tables';candidates.mkdir(exist_ok=True)
    with (ROOT/'materials/materials.csv').open() as stream:materials=list(csv.DictReader(stream))
    manifest=dict(source='NIST XrayMassCoef elemental tables',raw_sources={n:m for n,_,m in loaded},
                  generated_utc=datetime.now(timezone.utc).isoformat(),materials_sha256=digest(ROOT/'materials/materials.csv'),
                  generator_sha256=digest(Path(__file__)),physical_approval=False,method='mass-weighted elemental mu_en; approximate for compounds',
                  interpolation='log-log on each element, duplicate edge energies retained; exact edge uses right limit',materials=[])
    for material in materials:
        weights={el:float(material.get(f'{el}_wt_fraction',0)) for el in Z};weights={el:w for el,w in weights.items() if w}
        knots=sorted({r[0] for el in weights for r in tables[el]}|{0.6617,1.25,2.})
        edges={e for el in weights for e,n in Counter(r[0] for r in tables[el]).items() if n==2}
        # Dense samples limit interpolation error when BF interpolates a sum of powers.
        grid=sorted(set(knots)|{a*(b/a)**(j/8) for a,b in zip(knots,knots[1:]) for j in range(1,8)})
        rows=[]
        for energy in grid:
            if energy in edges:rows.append((energy,mixture(tables,weights,energy,'left')))
            rows.append((energy,mixture(tables,weights,energy)))
        path=candidates/f"{material['material_id']}.csv"
        write_csv(path,['Energy_MeV','mu_en_rho_cm2_g'],rows)
        manifest['materials'].append(dict(material_id=material['material_id'],material_definition_hash=material['material_definition_hash'],
            weights=weights,path=str(path.resolve()),sha256=digest(path),rows=len(rows),edges=len(edges),
            energy_min_MeV=rows[0][0],energy_max_MeV=rows[-1][0],physical_approval=False))
    for name in ('air','water','gadolinium','glass'):
        write_csv(out/f'reference_{name}.csv',['Energy_MeV','mu_en_rho_cm2_g'],[(r[0],r[2]) for r in parsed[name]])
    # Validation against true compound data measures the additivity approximation
    # for those controls only; it does not validate high-Z hafnates.
    controls={'water':{'H':0.111898,'O':0.888102},'air':{'C':0.000124,'N':0.755268,'O':0.231781,'Ar':0.012827},
              'gadolinium':{'O':0.084527,'S':0.084704,'Gd':0.830769},
              'glass':{'O':0.156453,'Si':0.080866,'Ti':0.008092,'As':0.002651,'Pb':0.751938}}
    comparisons=[]
    for name,weights in controls.items():
        rows=parsed[name];seen=Counter()
        for energy,_,reference in rows:
            side='left' if sum(r[0]==energy for r in rows)==2 and seen[energy]==0 else 'right';seen[energy]+=1
            estimate=mixture(tables,weights,energy,side)
            comparisons.append((name,energy,side,reference,estimate,100*(estimate/reference-1)))
    write_csv(out/'compound_controls.csv',['material','energy_MeV','side','NIST_mu_en','additive_mu_en','difference_pct'],comparisons)
    old=ROOT.parent/'BF/nist_mu_en_HEH1.csv'
    weights=next(m['weights'] for m in manifest['materials'] if m['material_id']=='HEH1')
    with old.open() as stream:legacy=list(csv.DictReader(stream))
    differences=[]
    for r in legacy:
        e=float(r['Energy_MeV']);value=float(r['mu_en_rho_cm2_g']);reference=mixture(tables,weights,e)
        differences.append((e,value,reference,100*(value/reference-1)))
    write_csv(out/'HEH1_legacy_nodes.csv',['energy_MeV','legacy','reconstructed_additive','legacy_difference_pct'],differences)
    manifest['legacy_HEH1_sha256']=digest(old)
    manifest['control_compositions']=controls
    manifest['control_composition_source']='https://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'raw_tables':len(loaded),'candidate_materials':len(materials),
        'HEH1_max_node_difference_pct':max(abs(r[-1]) for r in differences),
        'control_max_difference_pct':{n:max(abs(r[-1]) for r in comparisons if r[0]==n) for n in controls},
        'physical_approval':False},indent=2))

if __name__=='__main__':main()
