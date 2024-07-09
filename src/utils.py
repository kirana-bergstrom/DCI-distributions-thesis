import numpy as np


def distributeWeights(init_samples, bins_of_samples, w_bin, w_init=None, hist=False):

    n_bins = len(w_bin)
    w = np.empty(len(init_samples))

    if w_init is None:
        w_init = np.ones(len(init_samples))

    for i in range(n_bins):
        bin_inds = (bins_of_samples == i)
        n_i = np.sum(bin_inds)
        if hist:
            w[bin_inds] = w_bin[i] / n_i if n_i != 0 else 0
        else:
            w[bin_inds] = (w_bin[i] / n_bins) / np.sum(w_init[bin_inds]) if n_i != 0 else 0

    return w


def rejection_sampling(r):
    
    unifs = np.random.uniform(0,1,len(r))
    M = np.max(r)
    
    return (unifs < (r / M))


def edf(x, samples):

    n = len(samples)
    if len(np.shape(samples)) == 1:
        samples = np.reshape(samples, (n, 1))
        x = np.reshape(x, (1, 1))
    d = np.shape(samples)[1]

    which_lessthan = np.full(n, True)

    for dim in range(d):
        which_lessthan = [which_lessthan & (samples[:, dim] <= x[dim])]

    return (1 / n) * np.sum(which_lessthan)