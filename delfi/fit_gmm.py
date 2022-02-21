#!/usr/bin/env python

import numpy as np
import pygmmis
import sys

if __name__ == "__main__":
    assert len(sys.argv) == 3, "usage: fit.py <CNN result file> <GMM model file>"

    # load data from CNN
    # format: x,y,x',y'
    data = np.load(sys.argv[1])
    assert data.shape[1] == 4, "CNN resuls need to be in 4D"

    # run 5 GMMs and average them
    R = 5
    gmms = [ pygmmis.GMM(K=50, D=4) for _ in range(R) ]
    for r in range(5):
        pygmmis.fit(gmms[r], data, init_method='minmax', w=1e-2, cutoff=5, split_n_merge=3)
    gmm = pygmmis.stack(gmms, np.ones(R))

    # save GMM
    gmm.save(sys.argv[2])
