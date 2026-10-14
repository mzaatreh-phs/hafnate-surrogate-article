"""Atomic descriptors and an uncollided-source fluorescence baseline.

No Geant4 response label, fitted no-fluorescence term or measured scattered
source is used. This is a reduced model, not a validated physical replacement
for transport. Domain of the initial surrogate experiment: 0.1–0.2 MeV.
"""
from pathlib import Path
import sys
import time
import numpy as np
import xraylib as xr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import fluo_model as F
import radial_quadrature as Q

ENERGIES=[.1,.15,.2]
DEPTHS=[.5,1.,1.5,2.,2.5,3.,3.5,4.]
ATOMIC_FEATURES=['log_fluorescence_number_yield','fluorescence_energy_yield',
    'log_yield_weighted_attenuation','log_yield_harmonic_attenuation',
    'line_energy_mean_fraction','line_below_edge_fraction','line_edge_gap_mean_keV',
    'line_edge_gap_abs_mean_keV','Hf_Kalpha_attenuation_ratio','Hf_Kbeta_attenuation_ratio']


class Operator:
    def __init__(self,m,energy,depths,level):
        self.m,self.energy,self.depths=m,energy,depths
        primary=1000*energy
        self.lines=sorted(m.lines(primary),key=lambda r:(-round(r[2],4),r[0],r[1]))
        self.E=np.array([round(r[2],4) for r in self.lines])
        self.g=F._grid(depths,refinement=level,lower_radius=1e-6)
        self.mu0=m.mu(primary)
        self.a=np.array([m.mu(float(e))/self.mu0 for e in self.E])
        self.b=np.array([r[3]/self.mu0 for r in self.lines])
        lookup={(r[0],r[1]):i for i,r in enumerate(self.lines)}
        self.B=np.zeros((len(self.E),len(self.E)))
        for j,e in enumerate(self.E):
            for el,name,emitted,s in m.lines(float(e)):
                i=lookup[el,name]
                if self.E[i]>=e:
                    raise ValueError('atomic cascade is not strictly energy decreasing')
                self.B[i,j]+=s/self.mu0
        cache={}
        self.P=[]
        for a in self.a:
            if a not in cache:
                cache[a]=None if a>F.LOCAL_A else Q.propagation_matrix(self.g,a,16,64)
            self.P.append(cache[a])

    def solve(self):
        X=np.zeros((len(self.E),len(self.g)))
        for i,P in enumerate(self.P):
            source=self.b[i]*np.exp(-self.g)+self.B[i,:i]@X[:i]
            X[i]=source/self.a[i] if P is None else P@source
        return X

    def responses(self,X):
        result={}
        indices=np.array([np.flatnonzero(self.g==d)[0] for d in self.depths])
        for target,table in [('EBF',self.m.muen_air),('EABF',self.m.muen_mat)]:
            weight=self.E*np.array([F.interp_loglog(table,e/1000) for e in self.E])/(1000*self.energy*F.interp_loglog(table,self.energy))
            result[target]=1+weight@X[:,indices]*np.exp(self.depths)
        return result


def atomic(m,op):
    b,a,E=op.b,op.a,op.E
    number=sum(b)
    w=b/number
    cations=[e for e in m.w if e not in ('Hf','O')]
    cw=np.array([m.w[e] for e in cations]);cw/=sum(cw)
    edges=np.array([xr.EdgeEnergy(m.Z[e],xr.K_SHELL) for e in cations])
    gap=E[:,None]-edges[None,:]
    weights=w[:,None]*cw[None,:]
    def line_ratio(code):
        return m.mu(round(xr.LineEnergy(72,code),4))/op.mu0
    return dict(log_fluorescence_number_yield=float(np.log(number)),
        fluorescence_energy_yield=float(np.sum(b*E/(1000*op.energy))),
        log_yield_weighted_attenuation=float(np.log(w@a)),
        log_yield_harmonic_attenuation=float(np.log(1/(w@(1/a)))),
        line_energy_mean_fraction=float(w@E/(1000*op.energy)),
        line_below_edge_fraction=float(np.sum(weights*(gap<0))),
        line_edge_gap_mean_keV=float(np.sum(weights*gap)),
        line_edge_gap_abs_mean_keV=float(np.sum(weights*abs(gap))),
        Hf_Kalpha_attenuation_ratio=float(line_ratio(xr.KL3_LINE)),
        Hf_Kbeta_attenuation_ratio=float(line_ratio(xr.KM3_LINE)))


def material_features(mid,weights,energies=ENERGIES,depths=DEPTHS):
    start=time.perf_counter()
    m=F.Material(mid,weights)
    output=[];diagnostics=[]
    for energy in energies:
        previous=None
        for level in range(4):
            op=Operator(m,energy,depths,level)
            response=op.responses(op.solve())
            change=max(max(abs(response[q]-previous[q])/response[q]) for q in response) if previous is not None else None
            if change is not None and change<=.0005:break
            previous=response
            if level<3:del op
        if change>.0005:
            raise ArithmeticError(f'{mid} {energy}: spatial change {change} exceeds 0.05%')
        desc=atomic(m,op)
        for i,depth in enumerate(depths):
            output.append(dict(material_id=mid,energy_MeV=energy,depth_mfp=depth,
                               physics_EBF=float(response['EBF'][i]),physics_EABF=float(response['EABF'][i]),**desc))
        diagnostics.append(dict(energy_MeV=energy,refinement=level,nodes=len(op.g),lines=len(op.E),
                                adjacent_max_relative=change,lower_radius_MFP=1e-6,
                                regular_order=16,singular_order=64))
        del op
    return output,dict(material=mid,settings=diagnostics,wall_s=time.perf_counter()-start)
