from sklearn.neighbors import NearestNeighbors
import numpy as np
from geometry import *
import cv2

from operator import add

EPSILON = 0.01

def max_index(arr):
    """Returns the index of the maximum item in arr."""
    max_value = None
    max_index = None 
    for i in range(len(arr)):
        if max_value == None or arr[i] > max_value:
            max_value = arr[i]
            max_index = i

    return max_index

def cross_covariance(P, X, 
        P_nearest_neighbors = None,  # if we've already made a kd_tree, use that
        up = None, # fixed point on P, about which rotations occur
        ux = None):# fixed point on X, about which rotations occur
    """Returns the cross-covariance matrix of P and X using the 
    center of masses of each point cloud, as defined in [Besl1992].
    Asserts that P and X have same size."""
    assert len(P) == len(X)

    if P_nearest_neighbors == None:
        P_nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P)

    distances, indices = P_nearest_neighbors.kneighbors(X)

    # make an array of pairs (pi,xi) for each xi in X, such that
    # for xi, pi is the closest point to xi in P.
    matching = []
    for i in range(len(indices)):
        matching.append([P[indices[i][0]], X[i]])

    if up == None:
        up = center_of_mass(P)
    if ux == None:
        ux = center_of_mass(X)

    return reduce(add, [np.outer(px[1] - ux, px[0] - up) for px in matching]) / float(len(P))


def optimal_rotation_matrix(P, X, P_nearest_neighbors = None, up = None, ux = None):
    """Returns the optimal rotation matrix Q of the point set X for minimizing
    its covariance with P, a 3x3 matrix. See [Besl92] for details."""
    if P_nearest_neighbors == None:
        P_nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P)

    cross_covariance_matrix = cross_covariance(P,X, P_nearest_neighbors, up, ux)

    cc_transpose_difference = cross_covariance_matrix - cross_covariance_matrix.T
    Delta = np.array([cc_transpose_difference[1, 2], cc_transpose_difference[2, 0], cc_transpose_difference[0, 1]])

    trace           = np.trace(cross_covariance_matrix)
    transpose_trace = np.trace(cross_covariance_matrix.T)

    Q = np.zeros((4,4))
    Q[0,0]     = trace
    Q[0,1:4]   = Delta.T
    Q[1:4,0]   = Delta
    Q[1:4,1:4] = cross_covariance_matrix[0:3,0:3] + cross_covariance_matrix.T[0:3, 0:3] - trace * np.identity(3)

    v, w = np.linalg.eig(Q)
    max_eigenvalue_index = max_index(v)
    max_eigenvalue  = v[max_eigenvalue_index]
    max_eigenvector = w[:,max_eigenvalue_index]

    #print "Eigenvectors:"
    #print w
    #print "Eigenvalues:"
    #print v
    #print "Max Eigenvector:"
    #print max_eigenvector

    # this max_eigenvector is the rotation quaternion we need,
    # so make it a rotation matrix
    return promote(quaternion_to_rotation_matrix(unit_vector(max_eigenvector)))

def optimal_matricies(P, X, P_nearest_neighbors = None, up = None, ux = None):
    rot = optimal_rotation_matrix(P,X,P_nearest_neighbors,up,ux)
    return rot, translation_matrix(up - np.dot(rot, ux))

def mean_square_error(P,X):
    error = 0
    for i in range(len(P)):
        error += np.linalg.norm(P[i] - X[i], 2) # returns the l2 norm

    return error / len(P)

def icp(P,X,up = None, ux = None):
    """Returns the transformation matrix to take the point cloud X to the
    point cloud P by rigid transformation. If up and ux are specified, rotations
    and translations are relative to up on P and ux on X, which remain fixed in the
    transformation."""
    global_matrix = np.identity(4)
    X_copy = np.copy(X)

    if up == None:
        up = center_of_mass(P)
    if ux == None:
        ux = center_of_mass(X)

    P_nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P)

    last_error = mean_square_error(P,X_copy)
    for x in range(400):
        #print "Global Matrix:"
        #print global_matrix

        #print "X_copy:"
        #print X_copy

        rot, tr = optimal_matricies(P, X_copy, P_nearest_neighbors, up, ux)

        # for column vectors on right, rotate, then translate
        matrix = np.dot(tr, rot)

        global_matrix = np.dot(matrix, global_matrix)
        
        #print "Matrix:"
        #print matrix

        #print "P transformed:"
        #print apply_transform(matrix, P.copy())
        #print P.copy()

        X_copy = apply_transform(matrix, X_copy)
        ux = apply_transform(matrix, ux)

        assert(np.allclose(ux, up))

        # error should decrease with every step
        if last_error < 1e-8:
            break

        print last_error
        #assert(last_error >= mean_square_error(P,X_copy))
        last_error = mean_square_error(P,X_copy)


    return global_matrix
