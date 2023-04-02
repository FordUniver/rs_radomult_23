import sys
import os
import time
import pickle
from copy import deepcopy
from typing import Union, Iterable, Callable, Tuple, List, Dict, Any, Optional

import multiprocessing as mp
from tqdm import tqdm

import numpy as np
import itertools

from sage.all import *


#########################
### General utilities ###

def pbar(
        it:Iterable,
        total:Union[int, None]   = None,
        ncols:int                = 80,
        desc:Union[str, None]    = None,
        leave:bool               = True,
        verbose:bool             = True
    ) -> Iterable:

    if verbose:
        return tqdm(
            it,
            total      = total,
            ncols      = ncols,
            desc       = desc,
            leave      = leave,
            bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} ({elapsed}<{remaining})'
        )
    
    return it

def apply_pool(
        func:Callable, arguments:Iterable,
        pool:Union[mp.Pool, None]   = None,
        verbose:bool                     = True,
        desc:Union[str, None]            = None
    ) -> Iterable:

    arguments = list(arguments)
    if not isinstance(arguments[0], tuple): arguments = [(arg,) for arg in arguments]
    if verbose: arguments = pbar(arguments, leave=True, desc=desc, verbose=verbose)
    if pool is None: return [func(*arg) for arg in arguments]
    return pool.starmap(func, arguments)


#########################################################################
### Utilities to translate colorings from and to a hex representation ###

def hex_state_to_coloring(
        state:str, q:int, n:int,
        shift:Union[None, np.ndarray, list, tuple] = None
    ) -> np.ndarray:

    if shift is None: shift = np.array((0, )*n)
    else: shift = np.array(shift)

    state = bin(int(state, 16))[2:]
    state = '0'*(q**n - len(state)) + state

    coloring = np.zeros((q, )*n)

    for i, idx in enumerate(itertools.product(range(q), repeat=n)):
        coloring[tuple((np.array(idx)-shift)%q)] = int(state[i])

    return coloring


##########################
### Counting solutions ###

def count_solutions(coloring:np.ndarray, structure:str, q:int):
    Fqn = VectorSpace(GF(q), coloring.ndim)

    degenerate, nondegenerate = 0, 0
    dcount, ndcount = 0, 0

    if structure == 'Schur':

        for x_vec, y_vec in itertools.product(Fqn, Fqn):
            x = vec_to_idx(x_vec, q)
            y = vec_to_idx(y_vec, q)
            z = vec_to_idx(x_vec + y_vec, q)
            
            colors = set([coloring[x], coloring[y], coloring[z]])
            is_monochromatic = len(colors)==1
            
            if Matrix([x_vec, y_vec]).rank() == 2:
                ndcount += 1
                nondegenerate += is_monochromatic

            dcount += 1
            degenerate += is_monochromatic

    elif structure.endswith('AP'):
        k = int(structure.replace('AP', ''))
    
        for a, d in itertools.product(Fqn, Fqn):
            colors = set([coloring[vec_to_idx(a+j*d, q)] for j in range(k)])
            is_monochromatic = len(colors)==1 

            if not np.array_equal(d, Fqn[0]):
                ndcount += 1
                nondegenerate += is_monochromatic

            dcount += 1
            degenerate += is_monochromatic  

    else:
        raise NotImplementedError(structure)
 
    return QQ(degenerate / (dcount or 1)), QQ(nondegenerate / (ndcount or 1))


#############################################################
### Utilities for isomorphisms and densities of colorings ###

def arreq_in_list(myarr, list_arrays) -> bool:
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

def vec_to_idx(vec, q:int):
    temp = list(GF(q))
    return tuple([temp.index(x) for x in vec])
    
def isless(coloring1:np.ndarray, coloring2:np.ndarray, q:int, truncate:bool=False) -> bool:
    idxs1 = [vec_to_idx(vec, q) for vec in VectorSpace(GF(q), coloring1.ndim)]
    idxs2 = [vec_to_idx(vec, q) for vec in VectorSpace(GF(q), coloring2.ndim)]
    str1 = ''.join([str(coloring1[idx]) for idx in idxs1])
    str2 = ''.join([str(coloring2[idx]) for idx in idxs2])
    if truncate:
        str1 = str1[:min(len(str1), len(str2))]
        str2 = str2[:min(len(str1), len(str2))]
    return str1 < str2

def get_available_elements(coloring:np.ndarray, q:int, shift=None, basis=None) -> np.ndarray:
    basis = [] if basis is None else basis
    shift = VectorSpace(GF(q), coloring.ndim)[0] if shift is None else shift
    
    available_elements = np.ones_like(coloring, dtype=bool)
    
    for idx in VectorSpace(GF(q), len(basis)):
        ref_idx = shift + sum([i*b for i,b in zip(idx, basis)])
        available_elements[vec_to_idx(ref_idx, q)] = False
        
    return available_elements

def apply_transformation(coloring:np.ndarray, q:int, shift=None, basis=None) -> np.ndarray:
    basis = [] if basis is None else basis
    shift = VectorSpace(GF(q), coloring.ndim)[0] if shift is None else shift
    
    basis_dim = len(basis)
    
    transformed_coloring = np.zeros((q,)*basis_dim, dtype=coloring.dtype)
    
    for idx in VectorSpace(GF(q), basis_dim):
        ref_idx = shift + sum([i*b for i,b in zip(idx, basis)])
        transformed_coloring[vec_to_idx(idx, q)] = coloring[vec_to_idx(ref_idx, q)]
        
    return transformed_coloring

def canonize(
        coloring:np.ndarray, q:int, c:int,
        fixed_shift      = None,
        fixed_basis      = [],
        certificate:bool = False,
        invariant:bool   = False,
        canon_check:bool = False
    ) -> np.ndarray:
    
    if len(fixed_basis) > 0: assert fixed_shift is not None
    
    n = coloring.ndim
    Fqn = VectorSpace(GF(q), n)
    
    if canon_check: assert fixed_shift is None and len(fixed_basis)==0

    minc = np.min(coloring)
    
    if not invariant:
        current_leafs = [(coloring, Fqn[0], [], apply_transformation(coloring, q, shift=Fqn[0], basis=[]))]

    elif fixed_shift is None:
        current_leafs = [(coloring, shift, [], apply_transformation(coloring, q, shift=shift, basis=[])) for shift in Fqn if coloring[vec_to_idx(shift, q)]==minc]

    else:
        current_leafs = [(coloring, fixed_shift, fixed_basis, apply_transformation(coloring, q, shift=fixed_shift, basis=fixed_basis))]
    
    for _ in range(n-len(fixed_basis)):
        current_best = None
        
        next_leafs = []
        
        for coloring, shift, basis, transformed_coloring in current_leafs:
            available_elements = get_available_elements(coloring, q, shift=shift, basis=basis)
            
            for col in range(minc, c+2):
                candidates = [x for x in Fqn if coloring[vec_to_idx(x, q)] == col and available_elements[vec_to_idx(x, q)]]
                remainders = [x for x in Fqn if coloring[vec_to_idx(x, q)] > col and available_elements[vec_to_idx(x, q)]]
                if len(remainders) == 0: candidates = candidates[:1]
                if len(candidates) > 0: break
            
            for bvector in candidates:
                new_basis = basis + [bvector-shift]
                transformed_coloring = apply_transformation(coloring, q, shift=shift, basis=new_basis)
                if canon_check and isless(transformed_coloring, coloring, q, truncate=True): return False
                if current_best is None or isless(transformed_coloring, current_best, q): current_best = transformed_coloring
                next_leafs.append((coloring, shift, new_basis, transformed_coloring))
        
        current_leafs = []
        for coloring, shift, basis, transformed_coloring in next_leafs:
            if current_best is None or not isless(current_best, transformed_coloring, q):
                current_leafs.append((coloring, shift, basis, transformed_coloring))
        
    if canon_check:
        return True
    
    elif certificate:
        return current_leafs[0][3], current_leafs[0][1], current_leafs[0][2] 

    assert len(current_leafs) > 0
    assert len(current_leafs[0]) == 4

    return current_leafs[0][3]

def get_isomorphism_classes(
        q:int, n:int, c:int,
        invariant:bool                   = False,
        verbose:bool                     = True,
        store:bool                       = True,
        pool:Union[mp.Pool, None]        = None
    ) -> list:

    Fq = GF(q)
    Fqn = VectorSpace(Fq, n)

    current_leafs = [(c+1)*np.ones((q,)*n, dtype=int)]
    
    for idx in pbar(Fqn, total=q**n, leave=True, verbose=verbose):
        next_leafs = []

        potential_children = []
        for parent in current_leafs:
            for col in range(1, c+1):
                child = deepcopy(parent)
                child[vec_to_idx(idx, q)] = col
                potential_children.append(child)

        arguments = [(child, q, c, None, [], False, invariant, True) for child in potential_children]
        output = apply_pool(canonize, arguments, pool=pool, verbose=False)
        
        for child, is_canon in zip(potential_children, output):
            if is_canon: next_leafs.append(child)

        current_leafs = next_leafs

    return current_leafs

def get_pair_densities(coloring:np.ndarray, q:int, s:int, t:int, n:int, c:int, bases:list, invariant:bool=False) -> dict:
    Fq = GF(q)
    Fqn = VectorSpace(Fq, n)

    if s is None:
            
        assert invariant
            
        counts = {}
        total = 0

        for basis in bases:
            for selected_basis in itertools.combinations(basis, t):
                remaining_basis = [vec for vec in basis if not arreq_in_list(vec, selected_basis)]
                
                for a1, a2 in itertools.combinations(VectorSpace(Fq, n-t), 2):
                    shift1 = sum([a1[i]*r for i,r in enumerate(remaining_basis)])
                    shift2 = sum([a2[i]*r for i,r in enumerate(remaining_basis)])

                    flag1 = apply_transformation(coloring, q, shift=shift1, basis=selected_basis)
                    flag2 = apply_transformation(coloring, q, shift=shift2, basis=selected_basis)

                    cflag1 = canonize(flag1, q, c, invariant=invariant)
                    cflag2 = canonize(flag2, q, c, invariant=invariant)

                    if isless(cflag1, cflag2, q): key = (str(cflag1), str(cflag2))
                    else: key = (str(cflag2), str(cflag1))

                    counts[key] = counts.get(key, 0) + 1
                    total += 1
                    
        densities = {None: {key: QQ(val / (total*(2 if key[0]!=key[1] else 1))) for key, val in counts.items()}}
        
    else:
        counts = {}
        total = 0
        
        for basis in bases:
            for (shift, type_basis) in itertools.product([Fqn[0]] if not invariant else list(Fqn), itertools.combinations(basis, s)):
                ctype = apply_transformation(coloring, q, shift=shift, basis=type_basis)
                ctype, ctype_shift, ctype_basis = canonize(ctype, q, c, certificate=True, invariant=invariant)
                if str(ctype) not in counts: counts[str(ctype)] = {}
                
                fixed_shift = shift if s == 0 else shift + sum([x*b for x, b in zip(ctype_shift, type_basis)]) 
                fixed_basis = [sum([x*b for x, b in zip(cb, type_basis)]) for cb in ctype_basis]
                
                Fqt = VectorSpace(Fq, t)
                flag_fixed_shift = Fqt[0]
                flag_fixed_basis = [Fqt((Fq(0),)*i + (Fq(1),) + (Fq(0),)*(t-i-1)) for i in range(s)]
                
                remaining_basis = [vec for vec in basis if not arreq_in_list(vec, type_basis)]
                
                for basis1 in itertools.combinations(remaining_basis, t-s):
                    for basis2 in itertools.combinations([vec for vec in remaining_basis if not arreq_in_list(vec, basis1)], t-s):

                        flag1 = apply_transformation(coloring, q, shift=fixed_shift, basis=fixed_basis+list(basis1))
                        flag2 = apply_transformation(coloring, q, shift=fixed_shift, basis=fixed_basis+list(basis2))
                        
                        cflag1 = canonize(flag1, q, c, fixed_shift=flag_fixed_shift, fixed_basis=flag_fixed_basis, invariant=invariant)
                        cflag2 = canonize(flag2, q, c, fixed_shift=flag_fixed_shift, fixed_basis=flag_fixed_basis, invariant=invariant)
                        
                        if isless(cflag1, cflag2, q): key = (str(cflag1), str(cflag2))
                        else: key = (str(cflag2), str(cflag1))

                        counts[str(ctype)][key] = counts[str(ctype)].get(key, 0) + 1
                        total += 1
        
        densities = {key: {k: QQ(v / (total*(2 if k[0]!=k[1] else 1))) for k, v in val.items()} for key, val in counts.items()}
    
    return densities