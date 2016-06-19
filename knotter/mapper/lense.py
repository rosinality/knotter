import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.spatial.distance as dist
from sklearn import manifold

def pca(X, n_components=2):
    centered = X - X.mean(axis=0)

    U, s, Vt = la.svd(centered, full_matrices = False)

    s2 = s ** 2
    
    U = U[:, :n_components]
    s = s[:n_components]
    Vt = Vt[:n_components, :]
    
    return U, s, Vt, s2

def t_SNE(X, n_components=2):
    return manifold.TSNE(n_components=n_components, init='pca').fit_transform(X)

def spectral_embedding(X, n_components=2, n_neighbors=10):
    return manifold.SpectralEmbedding(
            n_components=n_components, n_neighbors=n_neighbors).fit_transform(X)

def Linfty_centering(X, options, metric='euclidean'):
    return dist.squareform(dist.pdist(X, metric=metric)).max(axis=0)

def pca_projection(X, n_axis = 2):
    N = X.shape[0]
    U, s, Vt, s2 = pca(X, n_axis)
    
    explained_variance = s2 / N

    # U * diag(s) is a score matrix.
    return U.dot(np.diag(s)), explained_variance / explained_variance.sum()

def simple_axis_projection(X, axis = 0):
    return X[:, axis]

def gaussian_density(X, options, metric='euclidean'):
    eps = float(options['epsilon'])
    dist_mat = dist.squareform(dist.pdist(X, metric=metric))
    
    return np.exp(-(dist_mat ** 2) / eps).sum(axis=0)

