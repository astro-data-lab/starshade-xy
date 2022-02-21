#!/usr/bin/env python

import numpy as np
import pygmmis
import sys

def split_gmm(gmm):
    # generate all marginal GMMs
    D = gmm.D // 2
    gmm_theta = pygmmis.GMM(K=gmm.K, D=D)
    gmm_t = pygmmis.GMM(K=gmm.K, D=D)
    gmm_x = pygmmis.GMM(K=gmm.K, D=D)

    # amplitudes are all the same
    gmm_theta.amp[:] = gmm_t.amp[:] = gmm_x.amp[:] = gmm.amp

    # subvector for means
    gmm_theta.mean[:, :] = gmm.mean[:, :D]
    gmm_t.mean[:, :] = gmm.mean[:, D:]
    gmm_x.mean[:, :] = 0 # not needed

    # submatrix for covariances:
    # actually C_t^-1 (needed often, store to avoid inverse)
    gmm_t.covar[:, :, :] = np.linalg.inv(gmm.covar[:, D:, D:])
    # actually C_x @ C_t^-1 (also needed often)
    gmm_x.covar[:, :, :] = gmm.covar[:, :D, D:] @ gmm_t.covar
    # C_theta|t = C_theta - C_x C_t^-1 C_x^t
    gmm_theta.covar[:, :, :] = gmm.covar[:, :D, :D] - gmm_x.covar @ gmm.covar[:, D:, :D]

    return gmm_theta, gmm_t, gmm_x

def gmm_theta_conditional(t, gmm_theta, gmm_t, gmm_x):
    K, D = gmm_theta.K, gmm_theta.D
    gmm_theta_ = pygmmis.GMM(K=K, D=D)

    # get mixture coefficients: evaluate gmm_t at location t
    delta_t = t - gmm_t.mean
    amp = np.log(gmm_t.amp) -0.5 * np.einsum('...i,...ij,...j', delta_t, gmm_t.covar, delta_t) + 0.5*np.log(np.linalg.det(gmm_t.covar)) - 0.5*D*np.log(2*np.pi)
    # renormalize mixing weights
    amp -= pygmmis.logsum(amp)
    gmm_theta_.amp[:] = np.exp(amp)

    # conditional mean
    gmm_theta_.mean[:, :] = gmm_theta.mean + np.einsum('...ij,...j', gmm_x.covar, delta_t)

    # conditional covariance is direct from C_theta
    gmm_theta_.covar[:, :, :] = gmm_theta.covar

    return gmm_theta_

def estimate_theta(gmm_theta, which="mean"):
    # mean gmm_theta
    mu_theta_ = np.sum(gmm_theta.amp[:,None] * gmm_theta.mean, axis=0)

    # in any case: report covariance of gmm_theta
    C_theta_ = np.sum(gmm_theta.amp[:,None,None] * ( gmm_theta.covar + (gmm_theta.mean - mu_theta_)[:,:,None] * (gmm_theta.mean - mu_theta_)[:,None,:]), axis=0)

    # report center of highest peak
    if which == "max":
        mu_theta_ = gmm_theta.mean[np.argmax(gmm_theta.amp / np.linalg.det(2*np.pi*gmm_theta.covar))]


    # report highest mode
    if which == "mode":
        raise NotImplementedError

    # sample from GMM
    if which == "sample":
        mu_theta_ = gmm_theta.draw(1)[0]


    return mu_theta_, C_theta_


if __name__ == "__main__":

    assert len(sys.argv) == 4, "usage: run_delfi.py <GMM model file> <CNN x> <CNN y>"

    # open GMM
    gmm = pygmmis.GMM.from_file(sys.argv[1])
    # split into precomputable sub-mixtures
    gmm_theta, gmm_t, gmm_x = split_gmm(gmm)

    # evaluate at p(theta | t_) for some measurement
    t_ = np.array([sys.argv[2], sys.argv[3]], dtype=np.float32)

    gmm_theta_ = gmm_theta_conditional(t_, gmm_theta, gmm_t, gmm_x)
    # use max (=approximate mode) for final estimated
    mu_theta_, C_theta_ = estimate_theta(gmm_theta_, which="max")

    print ("CNN estimate:\n", t_)
    print ("DELFI max:\n", mu_theta_)
    print ("DELFI covariance:\n", C_theta_)
