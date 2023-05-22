
"""
    CS131 HW0, Stanford 2022
"""

import numpy as np

def dot_product(a, b):
    out = None

    x, n = a.shape
    n, x = b.shape

    assert a.shape[1] == b.shape[0]
    out =  np.dot(a, b)
    assert out.shape == (x, x)
    return out



def eigen_decomp(M):
    w, v = np.linalg.eig(M)
    return w, v


def euclidean_distance_native(u, v):

    """
    I/P:
        u: A vector, represented as a list of floats.
        v: A vector, represented as a list of floats.

    O/P:
        float: Euclidean distance between 'u' and 'v'

    """

    assert isinstance(u, list)
    assert isinstance(v, list)

    s = 0
    res = 0

    for i in range(len(u)):
        s += (u[i] - v[i])**2

    res = s**0.5
    return res



def euclidean_distance_numpy(u, v):

    """
    I/P:
        u: A vector, represented as a Numpy array
        v: A vector, represented as a Numpy array

    O/P:
        float: Euclidean distance between 'u' and 'v'
    """
    assert isinstance(u, np.ndarray)
    assert isinstance(v, np.ndarray)
    assert u.shape == v.shape

    s = 0
    res = 0

    for i in range(len(u)):
        s += (u[i] - v[i])**2
        
    res = s**0.5
    
    return res



def get_eigen_values_and_vectors(M, k):

    """
    I/P:
        M: numpy matrix of shape (m, m)
        k: number of eigen vals and respective vectors to return.

    O/P:
        eigenvalues: list of length k containing the top k eigenvalues.
        eigenvectors: list of length k containing the top k eigenvectors.
    """
    
    eigenvalues = []
    eigenvectors = []
    w, v = np.linalg.eig(M)
    w_sorted_indx = np.argsort(w)
    for i in range(1, k+1):
        eigenvalues.append(w[w_sorted_indx[i]])
        eigenvectors.append(v[w_sorted_indx[i]])

    return eigenvalues, eigenvectors

    
