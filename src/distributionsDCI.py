import src.weightedEDFs as wedfs
import src.utils as utils
from sklearn.cluster import KMeans
import numpy as np
import scipy


def computePartitionedWeights_kMeans_decomposition(init_samples, pred_samples,
                                               sample_set_2=None, targ_CDF=None,
                                               n_clusters=None):

    n_samples = len(pred_samples)

    # if n_clusters is not input, then use n_clusters = n_samples / 100
    if n_clusters is None:
        n_clusters = n_samples / 100

    w = np.ones(n_samples)
    p_w = np.ones(n_samples)
    ps = []
    for i in range(np.shape(sample_set_2)[1]):
        p_w, _, _, _ = computePartitionedWeights_kMeans_IID(init_samples,
                                                      np.reshape(pred_samples[:,i], (n_samples,1)),
                                                      w_init=p_w,
                                                      sample_set_2=np.reshape(sample_set_2[:,i], (len(sample_set_2),1)),
                                                      n_clusters=n_clusters)
        ps.append(p_w)
        w *= p_w
    return w, ps


def computePartitionedWeights_regulargrid_decomposition(init_samples, pred_samples,
                                                    sample_set_2=None, targ_CDF=None,
                                                    n_clusters=None):

    n_samples = len(pred_samples)

    # if n_clusters is not input, then use n_clusters = n_samples / 100
    if n_clusters is None:
        n_clusters = n_samples / 100

    w = np.ones(n_samples)
    p_w = np.ones(n_samples)
    for i in range(np.shape(sample_set_2)[1]):
        p_w, _, _, _ = computePartitionedWeights_kMeans_IID(init_samples,
                                                      np.reshape(pred_samples[:,i], (n_samples,1)),
                                                      w_init=p_w,
                                                      sample_set_2=np.reshape(sample_set_2[:,i], (len(sample_set_2),1)),
                                                      n_clusters=n_clusters)

        w *= p_w
    return w


def computePartitionedWeights_regulargrid_nonIID(init_samples, pred_samples, des_samples, sample_set_2=None,
                                                 targ_CDF=None, n_bins=None, init_CDF=None):

    n_samples = len(pred_samples)
    m_samples = len(des_samples)
    dim_D = np.shape(pred_samples)[1]

    # if n_clusters is not input, then use n_clusters = n_samples / 100
    if n_bins is None:
        n_bins = n_samples / 100

    # create bins in each dimension
    low_ends = np.empty((n_bins, dim_D))
    upp_ends = np.empty((n_bins, dim_D))
    centers = np.empty((n_bins, dim_D))
    for d in range(dim_D):
        min_pred = np.min(pred_samples[:,d])
        max_pred = np.max(pred_samples[:,d])
        d_len = (max_pred - min_pred) / n_bins
        low_ends[:,d] = np.linspace(min_pred, max_pred-d_len, n_bins)
        upp_ends[:,d] = np.linspace(min_pred+d_len, max_pred, n_bins)
        centers[:,d] = np.linspace(min_pred+d_len/2, max_pred-d_len/2, n_bins)

    # ok for now this is only going to work in 1 or 2 dimensions because it's fucking recursive again!!!!!!
    # bin samples
    labels = np.empty(n_samples)
    if dim_D == 1:
        for i in range(n_bins):
            bin_inds = ((pred_samples[:,0] >= low_ends[i,0]) & (pred_samples[:,0] <= upp_ends[i,0]))
            labels[bin_inds] = i
    if dim_D == 2:
        counter = 0
        for i in range(n_bins):
            for j in range(n_bins):
                bin_inds = ((pred_samples[:,0] >= low_ends[i,0])
                            & (pred_samples[:,0] <= upp_ends[i,0])
                            & (pred_samples[:,1] >= low_ends[i,1])
                            & (pred_samples[:,1] <= upp_ends[i,1]))
                labels[bin_inds] = counter
                counter += 1

    # compute weights for clusters
    H_bin = wedfs.compute_H(centers)
    if targ_CDF is not None:
        b_bin = wedfs.compute_b(centers, targ_CDF=targ_CDF)
    elif sample_set_2 is not None:
        b_bin = wedfs.compute_b(centers, sample_set_2=sample_set_2)
    w_bin = wedfs.compute_optimal_w(H_bin, b_bin)

    # compute weights on input
    H_init = wedfs.compute_H(init_samples)
    b_init = wedfs.compute_b(init_samples, sample_set_2=des_samples)
    w_init = wedfs.compute_optimal_w(H_init, b_init)

    # distribute weights in clusters
    w = utils.distributeWeights(init_samples, labels, w_bin, w_init=w_init)

    return w * w_init[:,0], labels, w_init[:,0], w


def computePartitionedWeights_kMeans_nonIID(init_samples, pred_samples, des_samples,
                                            sample_set_2=None, targ_CDF=None, n_clusters=None):

    n_samples = len(pred_samples)
    m_samples = len(des_samples)

    # if n_clusters is not input, then use n_clusters = n_samples / 100
    if n_clusters is None:
        n_clusters = n_samples / 100

    # cluster weights using kMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pred_samples)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # compute weights for clusters
    H_cluster = wedfs.compute_H(centers)
    if targ_CDF is not None:
        b_cluster = wedfs.compute_b(centers, targ_CDF=targ_CDF)
    elif sample_set_2 is not None:
        b_cluster = wedfs.compute_b(centers, sample_set_2=des_samples)
    w_cluster = wedfs.compute_optimal_w(H_cluster, b_cluster)

    # compute weights on input
    H_init = wedfs.compute_H(init_samples)
    b_init = wedfs.compute_b(init_samples, sample_set_2=des_samples)
    w_init = wedfs.compute_optimal_w(H_init, b_init)

    # distribute weights in clusters
    w = utils.distributeWeights(init_samples, labels, w_cluster, w_init=w_init)

    return w * w_init[:,0], labels, w_init[:,0], w


def computePartitionedWeights_kMeans_IID(init_samples, pred_samples, w_init=None,
                                         sample_set_2=None, targ_CDF=None, n_clusters=None):

    n_samples = len(pred_samples)

    # if n_clusters is not input, then use n_clusters = n_samples / 100
    if n_clusters is None:
        n_clusters = n_samples / 100

    # cluster weights using kMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pred_samples)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # compute weights for clusters
    H_cluster = wedfs.compute_H(centers)
    if targ_CDF is not None:
        b_cluster = wedfs.compute_b(centers, targ_CDF=targ_CDF)
    elif sample_set_2 is not None:
        b_cluster = wedfs.compute_b(centers, sample_set_2=sample_set_2)
    w_cluster = wedfs.compute_optimal_w(H_cluster, b_cluster)
    
    # distribute weights evenly in clusters
    w = utils.distributeWeights(init_samples, labels, w_cluster, w_init=w_init)

    return w, labels, centers, w_cluster


def computePartitionedWeights_regulargrid_IID(init_samples, pred_samples, bbox=None,
                                              sample_set_2=None, targ_CDF=None, n_bins=None,
                                              remove_empty_bins=True):

    n_samples = len(pred_samples)
    dim_D = np.shape(pred_samples)[1]

    if n_bins is None: # idk this isn't going to work well in multiple dimensions
        n_bins = (n_samples / 100)
    if isinstance(n_bins, int):
        n_bins = [n_bins] * dim_D

    if bbox is not None:
        H, edges = np.histogramdd(pred_samples, bins=n_bins, range=bbox)
    else:
        H, edges = np.histogramdd(pred_samples, bins=n_bins)

    low_ends = np.empty((np.prod(n_bins), dim_D))
    upp_ends = np.empty((np.prod(n_bins), dim_D))

    if dim_D == 1:
        low_ends = edges[0][:-1]
        upp_ends = edges[0][1:]
    if dim_D == 2:
        for i in range(n_bins[0]):
            for j in range(n_bins[1]):
                low_ends[i*n_bins[1]+j, 0] = edges[0][:-1][i]
                upp_ends[i*n_bins[1]+j, 0] = edges[0][1:][i]
                low_ends[i*n_bins[1]+j, 1] = edges[1][:-1][j]
                upp_ends[i*n_bins[1]+j, 1] = edges[1][1:][j]
    if dim_D == 3:
        for i in range(n_bins[0]):
            for j in range(n_bins[1]):
                for k in range(n_bins[2]):
                    low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k, 0] = edges[0][:-1][i]
                    upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k, 0] = edges[0][1:][i]
                    low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k, 1] = edges[1][:-1][j]
                    upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k, 1] = edges[1][1:][j]
                    low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k, 2] = edges[2][:-1][k]
                    upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k, 2] = edges[2][1:][k]

    labels = np.empty(n_samples)
    centers = []
    if dim_D == 1:
        counter = 0
        for i in range(n_bins[0]):
            bin_inds = ((pred_samples[:,0] >= low_ends[i]) & (pred_samples[:,0] <= upp_ends[i]))
            labels[bin_inds] = counter
            if remove_empty_bins and np.sum(bin_inds) != 0:
                centers.append([(upp_ends[i] - low_ends[i]) / 2 + low_ends[i]])
                counter += 1
            else:
                centers.append([(upp_ends[i] - low_ends[i]) / 2 + low_ends[i]])
                counter += 1
    if dim_D == 2:
        counter = 0
        for i in range(n_bins[0]):
            for j in range(n_bins[1]):
                bin_inds = ((pred_samples[:,0] >= low_ends[i*n_bins[1]+j,0])
                            & (pred_samples[:,0] <= upp_ends[i*n_bins[1]+j,0])
                            & (pred_samples[:,1] >= low_ends[i*n_bins[1]+j,1])
                            & (pred_samples[:,1] <= upp_ends[i*n_bins[1]+j,1]))
                labels[bin_inds] = counter
                if remove_empty_bins and np.sum(bin_inds) != 0:
                    centers.append([(upp_ends[i*n_bins[1]+j,0] - low_ends[i*n_bins[1]+j,0]) / 2  + low_ends[i*n_bins[1]+j,0],
                                    (upp_ends[i*n_bins[1]+j,1] - low_ends[i*n_bins[1]+j,1]) / 2 + low_ends[i*n_bins[1]+j,1]])
                    counter += 1
                else:
                    centers.append([(upp_ends[i*n_bins[1]+j,0] - low_ends[i*n_bins[1]+j,0]) / 2  + low_ends[i*n_bins[1]+j,0],
                                    (upp_ends[i*n_bins[1]+j,1] - low_ends[i*n_bins[1]+j,1]) / 2 + low_ends[i*n_bins[1]+j,1]])
                    counter += 1
    if dim_D == 3:
        counter = 0
        for i in range(n_bins[0]):
            for j in range(n_bins[1]):
                for k in range(n_bins[2]):
                    bin_inds = ((pred_samples[:,0] >= low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,0])
                                & (pred_samples[:,0] <= upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,0])
                                & (pred_samples[:,1] >= low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,1])
                                & (pred_samples[:,1] <= upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,1])
                                & (pred_samples[:,2] >= low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,2])
                                & (pred_samples[:,2] <= upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,2]))
                    labels[bin_inds] = counter
                    if remove_empty_bins and np.sum(bin_inds) != 0:
                        centers.append([(upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,0] - low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,0]) / 2 + low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,0],
                                        (upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,1] - low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,1]) / 2 + low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,1],
                                        (upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,2] - low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,2]) / 2 + low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,2]])
                        counter += 1
                    else:
                        centers.append([(upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,0] - low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,0]) / 2 + low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,0],
                                        (upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,1] - low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,1]) / 2 + low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,1],
                                        (upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,2] - low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,2]) / 2 + low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,2]])
                        counter += 1
                        
    if dim_D > 3:
        print('ERROR')
    
    centers = np.array(centers)

    # compute weights for clusters
    H_bin = wedfs.compute_H(centers)
    if targ_CDF is not None:
        b_bin = wedfs.compute_b(centers, targ_CDF=targ_CDF)
    elif sample_set_2 is not None:
        b_bin = wedfs.compute_b(centers, sample_set_2=sample_set_2)
    w_bin = wedfs.compute_optimal_w(H_bin, b_bin)
    
    # distribute weights evenly in clusters
    w = utils.distributeWeights(init_samples, labels, w_bin)

    return w, labels, centers, w_bin


def computeHistogramWeights_kMeans_IID(init_samples, pred_samples, w_init=None,
                                       sample_set_2=None, n_clusters=None):

    n_samples = len(pred_samples)
    n_obs = len(sample_set_2)

    # if n_clusters is not input, then use n_clusters = n_samples / 100
    if n_clusters is None:
        n_clusters = n_samples / 100

    # cluster weights using kMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pred_samples)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    labels_obs = kmeans.predict(sample_set_2)

    ratios = []
    for bin_id in range(n_clusters):
        ratios.append(np.sum([labels_obs == bin_id])/n_obs)
    
    # distribute weights evenly in clusters
    w = utils.distributeWeights(init_samples, labels, ratios, hist=True)

    return w, labels, ratios


def computeHistogramWeights_regulargrid_IID(init_samples, pred_samples, bbox=None,
                                              sample_set_2=None, n_bins=None,
                                              remove_empty_bins=True):

    n_samples = len(pred_samples)
    dim_D = np.shape(pred_samples)[1]
    n_obs = len(sample_set_2)

    if n_bins is None: # idk this isn't going to work well in multiple dimensions
        n_bins = (n_samples / 100)
    if isinstance(n_bins, int):
        n_bins = [n_bins] * dim_D

    if bbox is not None:
        H, edges = np.histogramdd(pred_samples, bins=n_bins, range=bbox)
    else:
        H, edges = np.histogramdd(pred_samples, bins=n_bins)

    low_ends = np.empty((np.prod(n_bins), dim_D))
    upp_ends = np.empty((np.prod(n_bins), dim_D))

    if dim_D == 1:
        low_ends = edges[0][:-1]
        upp_ends = edges[0][1:]
    if dim_D == 2:
        for i in range(n_bins[0]):
            for j in range(n_bins[1]):
                low_ends[i*n_bins[1]+j, 0] = edges[0][:-1][i]
                upp_ends[i*n_bins[1]+j, 0] = edges[0][1:][i]
                low_ends[i*n_bins[1]+j, 1] = edges[1][:-1][j]
                upp_ends[i*n_bins[1]+j, 1] = edges[1][1:][j]
    if dim_D == 3:
        for i in range(n_bins[0]):
            for j in range(n_bins[1]):
                for k in range(n_bins[2]):
                    low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k, 0] = edges[0][:-1][i]
                    upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k, 0] = edges[0][1:][i]
                    low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k, 1] = edges[1][:-1][j]
                    upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k, 1] = edges[1][1:][j]
                    low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k, 2] = edges[2][:-1][k]
                    upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k, 2] = edges[2][1:][k]

    labels = np.empty(n_samples)
    ratios = []
    if dim_D == 1:
        counter = 0
        for i in range(n_bins[0]):
            bin_inds = ((pred_samples[:,0] >= low_ends[i]) & (pred_samples[:,0] <= upp_ends[i]))
            labels[bin_inds] = counter
            if remove_empty_bins and np.sum(bin_inds) != 0:
                ratios.append(np.sum((sample_set_2[:,0] >= low_ends[i]) & (sample_set_2[:,0] <= upp_ends[i]))/n_obs)
                counter += 1
            else:
                ratios.append(np.sum((sample_set_2[:,0] >= low_ends[i]) & (sample_set_2[:,0] <= upp_ends[i]))/n_obs)
                counter += 1
    if dim_D == 2:
        counter = 0
        for i in range(n_bins[0]):
            for j in range(n_bins[1]):
                bin_inds = ((pred_samples[:,0] >= low_ends[i*n_bins[1]+j,0])
                            & (pred_samples[:,0] <= upp_ends[i*n_bins[1]+j,0])
                            & (pred_samples[:,1] >= low_ends[i*n_bins[1]+j,1])
                            & (pred_samples[:,1] <= upp_ends[i*n_bins[1]+j,1]))
                labels[bin_inds] = counter
                if remove_empty_bins and np.sum(bin_inds) != 0:
                    ratios.append(np.sum((sample_set_2[:,0] >= low_ends[i,0])
                                         & (sample_set_2[:,0] <= upp_ends[i,0])
                                         & (sample_set_2[:,1] >= low_ends[i,1])
                                         & (sample_set_2[:,1] <= upp_ends[i,1])
                                        )/n_obs)
                    counter += 1
                else:
                    ratios.append(np.sum((sample_set_2[:,0] >= low_ends[i,0])
                                         & (sample_set_2[:,0] <= upp_ends[i,0])
                                         & (sample_set_2[:,1] >= low_ends[i,1])
                                         & (sample_set_2[:,1] <= upp_ends[i,1])
                                        )/n_obs)
                    counter += 1
    if dim_D == 3:
        counter = 0
        for i in range(n_bins[0]):
            for j in range(n_bins[1]):
                for k in range(n_bins[2]):
                    bin_inds = ((pred_samples[:,0] >= low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,0])
                                & (pred_samples[:,0] <= upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,0])
                                & (pred_samples[:,1] >= low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,1])
                                & (pred_samples[:,1] <= upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,1])
                                & (pred_samples[:,2] >= low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,2])
                                & (pred_samples[:,2] <= upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,2]))
                    labels[bin_inds] = counter
                    if remove_empty_bins and np.sum(bin_inds) != 0:
                        ratios.append(np.sum((sample_set_2[:,0] >= low_ends[i,0])
                                             & (sample_set_2[:,0] <= upp_ends[i,0])
                                             & (sample_set_2[:,1] >= low_ends[i,1])
                                             & (sample_set_2[:,1] <= upp_ends[i,0])
                                             & (sample_set_2[:,2] >= low_ends[i,2])
                                             & (sample_set_2[:,2] <= upp_ends[i,2])
                                            )/n_obs)
                        counter += 1
                    else:
                        ratios.append(np.sum((sample_set_2[:,0] >= low_ends[i,0])
                                             & (sample_set_2[:,0] <= upp_ends[i,0])
                                             & (sample_set_2[:,1] >= low_ends[i,1])
                                             & (sample_set_2[:,1] <= upp_ends[i,0])
                                             & (sample_set_2[:,2] >= low_ends[i,2])
                                             & (sample_set_2[:,2] <= upp_ends[i,2])
                                            )/n_obs)
                        counter += 1
                        
    if dim_D > 3:
        print('ERROR')

    w = utils.distributeWeights(init_samples, labels, ratios, hist=True)

    return w, labels, ratios