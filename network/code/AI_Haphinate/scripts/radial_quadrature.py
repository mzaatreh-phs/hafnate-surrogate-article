"""Cellwise quadrature for the fluorescence point kernel on a linear grid.

Sources are interpolated linearly inside each cell. A square substitution
handles the logarithmic kernel singularity in cells touching the scoring node.
No absolute-fluence tolerance is used. This integrates the existing kernel;
spatial interpolation and atomic-physics approximations remain separate.
"""
from functools import lru_cache
import numpy as np
from scipy.special import exp1, roots_legendre


def _validate(grid, a):
    g = np.asarray(grid, dtype=float)
    if g.ndim != 1 or len(g)<2 or not np.all(np.isfinite(g)) or g[0]<=0 or np.any(np.diff(g)<=0):
        raise ValueError('radial grid must be finite, positive and strictly increasing')
    if not np.isfinite(a) or a<=0:
        raise ValueError('attenuation ratio must be finite and positive')
    return g


def propagation_matrix(grid, a, regular_order=8, singular_order=32):
    """Integrate all linear hat functions, using higher order near each node."""
    g = _validate(grid, a)
    if regular_order<2 or singular_order<regular_order:
        raise ValueError('quadrature orders must satisfy 2 <= regular <= singular')
    zr, wr = roots_legendre(regular_order)
    zs, ws = roots_legendre(singular_order)
    ur, wr = (zr+1)/2, wr/2
    us, ws = (zs+1)/2, ws/2
    lo, hi = g[:-1], g[1:]
    h = hi-lo
    regular_nodes = lo[:,None]+h[:,None]*ur
    P = np.zeros((len(g),len(g)))
    for i,tau in enumerate(g):
        distance = np.maximum(np.maximum(lo-tau,tau-hi),0)
        near = distance <= 2*h
        t = regular_nodes
        kernel = tau/(2*t)*(exp1(a*np.abs(tau-t))-exp1(a*(tau+t)))
        val = kernel*h[:,None]*wr
        left = np.sum(val*(1-ur),axis=1)
        right = np.sum(val*ur,axis=1)
        ids = np.flatnonzero(near)
        t = lo[ids,None]+h[ids,None]*us
        jac = np.broadcast_to(h[ids,None],t.shape).copy()
        for j,cell in enumerate(ids):
            if cell==i-1:
                t[j] = tau-h[cell]*us*us
                jac[j] = 2*h[cell]*us
            elif cell==i:
                t[j] = tau+h[cell]*us*us
                jac[j] = 2*h[cell]*us
        kernel = tau/(2*t)*(exp1(a*np.abs(tau-t))-exp1(a*(tau+t)))
        val = kernel*jac*ws
        frac = (t-lo[ids,None])/h[ids,None]
        left[ids] = np.sum(val*(1-frac),axis=1)
        right[ids] = np.sum(val*frac,axis=1)
        P[i,:-1] += left
        P[i,1:] += right
    if not np.all(np.isfinite(P)) or np.min(P)<-1e-13:
        raise ArithmeticError('invalid radial quadrature matrix')
    return np.maximum(P,0)


@lru_cache(maxsize=48)
def _cached_matrix(nodes, a, regular_order, singular_order):
    P = propagation_matrix(np.asarray(nodes),a,regular_order,singular_order)
    P.flags.writeable = False
    return P


def propagate(grid, source, a, regular_order=8, singular_order=32):
    g = _validate(grid,a)
    source = np.asarray(source,dtype=float)
    if source.shape != g.shape or not np.all(np.isfinite(source)):
        raise ValueError('source must be finite and match the radial grid')
    return _cached_matrix(tuple(g),float(a),regular_order,singular_order) @ source


def clear_cache():
    _cached_matrix.cache_clear()
