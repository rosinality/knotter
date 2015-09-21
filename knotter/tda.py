import asyncio
import threading
from aiohttp import web
import json
import pandas as pd
from io import StringIO
import numpy as np
import scipy as sp
import scipy.linalg as la
from scipy.cluster import hierarchy
import scipy.spatial.distance as dist
from scipy import stats
import itertools
import functools

#def overlapping_interval(minimum, maximum, N, overlap = .5):
#    delta = overlap / 2
#    interval = np.linspace(minimum, maximum, N)
    
    # pre = [1, 2, 3] - (overlap / 2)
#    pre = interval - delta
    
    # post = [1, 2, 3] + (overlap / 2) + (M - m + 1) / N
    # (M - m + 1) N = interval between 2 consecutive numbers.
#    post = interval + delta + (maximum - minimum + 1) / N
    
#    return np.dstack((pre, post))[0]

def overlapping_interval(minimum, maximum, N, overlap = .5):
    interval = (maximum - minimum) / (N - (N - 1) * overlap)
    step = interval * (1 - overlap)
    intervals = []
    
    for i in range(N):
        interval_min = minimum + i * step
        interval_max = interval_min + interval
        intervals.append([interval_min, interval_max])
    
    return np.vstack(intervals)

def pca(X, n_components = 2):
    centered = X - X.mean(axis=0)

    U, s, Vt = la.svd(centered, full_matrices = False)

    s2 = s ** 2
    
    U = U[:, :n_components]
    s = s[:n_components]
    Vt = Vt[:n_components, :]
    
    return U, s, Vt, s2

def Linfty_centering(X, options):
    return dist.squareform(dist.pdist(X)).max(axis=0)

def pca_projection(X, n_axis = 2):
    N = X.shape[0]
    U, s, Vt, s2 = pca(X, n_axis)
    
    explained_variance = s2 / N

    # U * diag(s) is score matrix.
    return U.dot(np.diag(s)), explained_variance / explained_variance.sum()

def simple_axis_projection(X, axis = 0):
    return X[:, axis]

def gaussian_density(X, options):
    eps = float(options['epsilon'])
    dist_mat = dist.squareform(dist.pdist(X))
    
    return np.exp(-(dist_mat ** 2) / eps).sum(axis=0)

def point_in_interval(refracted, interval):
    result = []
    
    for i in interval:
        # appends *index* of points in the interval.
        result.append(np.where((i[0] <= refracted) & (refracted < i[1]))[0])
        
    return result

def point_in_intervals(refracted, intervals):
    # Example:
    # refracted -> array([[v1, v2, ...], [v3, v4, ...], ...])
    # intervals -> [np.array([[i1, i2], [i3, i4], ...]),
    #               np.array([[j1, j2], ...]), ...]
    # if nth refracted value vector [v1, v2, ...] (output of lense function)
    # is in the box of intervals [i1, i2] X [j1, j2] X ...,
    # then append n to the list, and this function returns aggregation of
    # such lists.
    # this function considers order of such boxes with points are not
    # significant.
    result = []
    
    for interval in itertools.product(*intervals):
        subsquare = []
        for no, r in enumerate(refracted):
            contained = True
            for i, v in enumerate(r):
                if not (interval[i][0] <= v < interval[i][1]):
                    contained = False
                    break
                    
            if contained:
                subsquare.append(no)
        result.append(np.array(subsquare))
        
    return result

def index_to_points(X, index):
    size = index.shape[0]
    N = X.shape[1]
    part = np.zeros((size, N))
    
    for i, idx in enumerate(index):
        part[i] = X[idx]
    
    return part

#def make_cluster(X, index, eps = .1):
#    Z = hierarchy.linkage(X)
    # this returns array like [3, 1, 2, 1, 1, 2]
    # each number is index of cluster, and index of each element is
    # the index of elements in given input(data).
#    cutoff = hierarchy.fcluster(Z, eps, criterion = 'distance')
#    clusters = []
    
#    for i in range(np.max(cutoff)):
#        clusters.append(set())
        
#    for i, cluster_no in enumerate(cutoff):
#        clusters[cluster_no - 1].add(index[i])
        
    # ex: [{1, 2, 3}, {2, 3, 4}, {4, 5}, {6, 7, 9}]
#    return clusters

def make_cluster(X, points):
    clusters = []
    
    for p in points:
        if p.shape[0] > 1:
            #clusters.append(hierarchy.linkage(index_to_points(X, p)))
            clusters.append(hierarchy.linkage(
                sp.spatial.distance.pdist(index_to_points(X, p))))
            
        elif p.shape[0] == 1:
            clusters.append(np.array([]))
            
    return clusters

def cut_points(points):
    result = []
    
    for p in points:
        if p.shape[0] >= 1:
            result.append(p)
            
    return result

def make_cluster2(X, points):
    clusters = []
    
    for p in points:
        if p.shape[0] >= 1:
            clusters.append(hierarchy.linkage(index_to_points(X, p)))
            
    return clusters

def _cutoff_threshold(distance, max_distance, threshold):
    return max_distance * threshold
    
def cutoff_threshold(threshold):
    return functools.partial(_cutoff_threshold, threshold = threshold)

def _cutoff_histogram(distance, max_distance, bins, nth):
    hist, edge = np.histogram(distance, bins)
    zero, = np.nonzero(hist)
    
    if zero.shape[0] != 0:
        return edge[zero[nth]]
        
    else:
        return max_distance
    
def cutoff_histogram(bins, nth=0):
    return functools.partial(_cutoff_histogram, bins=bins, nth=nth)

def _cutoff_histogram_max_gap(distance, max_distance, bins):
    hist, edge = np.histogram(distance, bins)
    return 0

def cutoff_histogram_max_gap(bins):
    return functools.partial(_cutoff_histogram, bins = bins)

def cut_cluster(clusters, points, threshold = .1):
    distances = cluster_distance(clusters)
    #cutoff = np.percentile(distances, threshold * 100)
    #cutoff = threshold
    result = []
    
    for no, c in enumerate(clusters):
        if c.shape[0] < 1:
            result.append(set(points[no]))
            
        else:
            #cutoff = c[-1, 2] * threshold
            cutoff = threshold(c[:, 2], c[-1, 2])
            #print(cutoff)
            
            cluster_index = hierarchy.fcluster(c, cutoff,
                    criterion = 'distance')
            point_clusters = []

            for i in range(np.max(cluster_index)):
                point_clusters.append(set())

            for i, cluster_no in enumerate(cluster_index):
                point_clusters[cluster_no - 1].add(points[no][i])

            result.extend(point_clusters)
                
    return result

def cluster_distance(clusters):
    distances = np.array([])
    
    for c in clusters:
        try:
            distances = np.append(distances, c[:, 2])
            
        except:
            continue
            
    return distances

def make_all_cluster(X, points, eps = .1):
    clusters = []
    
    for p in points:
        if p.shape[0] < 1:
            continue
        elif p.shape[0] < 2:
            clusters.append(set([p[0]]))
        else:
            clusters.extend(make_cluster(index_to_points(X, p), p, eps))
        
    return clusters

def find_nerves(clusters, threshold = 1):
    links = []
    
    if threshold == 1:
        for i, j in itertools.combinations(range(len(clusters)), 2):
            if not clusters[i].isdisjoint(clusters[j]):
                links.append((i, j))
        
    else:
        for i, j in itertools.combinations(range(len(clusters)), 2):
            if len(clusters[i].intersection(clusters[j])) >= threshold:
                links.append((i, j))
                
    return links

def cluster_size(clusters):
    result = np.empty(len(clusters))
    
    for n, c in enumerate(clusters):
        result[n] = len(c)
        
    return result

def cluster_coloring(clusters, data, method = 'mean'):
    result = []
    
    if method == 'mean':
        f = np.mean
    elif method == 'median':
        f = np.median
    elif method == 'mode':
        f = lambda x: stats.mode(x)[0][0]
    
    for c in clusters:
        elem = list(c)
        v = []
        for e in elem:
            v.append(data[e])
        result.append(f(v))
        
    return result

def lense_from_cluster(clusters, lense, nth=0):
    result = []

    for c in clusters:
        lense_values = []
        
        for i in c:
            lense_values.append(lense[i, nth])

        result.append(lense_values)

    return result

def mean_lense(lense_values):
    result = []
    
    for l in lense_values:
        result.append(np.nan_to_num(np.nanmean(l)))

    return result

