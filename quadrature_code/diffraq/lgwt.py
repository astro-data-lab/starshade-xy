"""
lgwt.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Gauss-Legendre quadrature scheme on a 1D interval.
    Taken from FRESNAQ's lgwt.m (Barnett 2021).

"""

import numpy as np

def lgwt(N, a, b):
    """
    pq, wq = lgwt(N, a, b)

    computes the N-point Legendre-Gauss nodes p and weights w on a 1D interval [a,b].

    Inputs:
        N = # node points
        a = lower bound
        b = upper bound

    Outputs:
        pq = numpy array of nodes
        wq = numpy array of weights
    """

    # return np.linspace(a, b, N), np.ones(N)/N
    #Use numpy legendre (isn't tested above 100, but agrees for me)

    #Get nodes, weights on [-1,1]
    p1, wq = np.polynomial.legendre.leggauss(N)
    p1 = p1[::-1]
    wq = wq[::-1]

    #Linear map from [-1,1] to [a,b]
    pq = (a*(1-p1) + b*(1+p1))/2

    #Normalize the weights
    wq *= (b-a)/2

    return pq, wq
