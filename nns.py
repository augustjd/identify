from sklearn.neighbors import NearestNeighbors
import numpy as np

from operator import add

def max_index(arr):
    """Returns the index of the maximum item in arr."""
    max = None
    max_index = None for i in range(len(arr)):
        if max == None or arr[i] > max:
            max = arr[i]
            max_index = i

    return max_index

def nearest_neighbor(p, A, A_nearest_neighbors = None):
    """Returns the point in A that is nearest to p, as a k-dimensional numpy.array()"""
    if A_nearest_neighbors == None:
        A_nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(A)

    dist, indices = A_nearest_neighbors.kneighbors(p)
    # dist is easily recoverable from p and the closest point in A, so just give the closest point
    return A[indices[0][0]]

def center_of_mass(P):
    """Returns the 'center of mass', or average point,
    in the point cloud P."""
    return reduce(add, P) / float(len(P))

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

    matching = [ [P[i], X[index_arr[0]]] for i,index_arr in enumerate(indices)]

    print matching

    if up == None:
        up = center_of_mass(P)
    if ux == None:
        ux = center_of_mass(X)

    return reduce(add, [np.outer(px[0] - up, px[1] - ux) for px in matching]) / float(len(P))

def translation_matrix(x,y,z):
    return np.array([[1,0,0,x], 
                     [0,1,0,y], 
                     [0,0,1,z],
                     [0,0,0,1]])

def quaternion_to_rotation_matrix(q):
    """Returns a 4x4 matrix for rotation in R3, with additional zeroed final column and row,
    for composability with translation matricies."""
    return np.array([
        [pow(q[0],2) + pow(q[1],2) - pow(q[2],2) - pow(q[3],2), 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2]),0],
        [2*(q[1]*q[2] + q[0]*q[3]), pow(q[0],2) + pow(q[2],2) - pow(q[1],2) - pow(q[3],2), 2*(q[2]*q[3] - q[0]*q[1]),0],
        [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), pow(q[0],2) + pow(q[3],2) - pow(q[1],2) - pow(q[2],2)],0],
        [0,0,0,0])

def optimal_rotation_matrix(P, X, P_nearest_neighbors = None, up = None, ux = None):
    """Finds the optimal rotation matrix Q of the point sets P and X, assuming P is fixed."""
    if P_nearest_neighbors == None:
        P_nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P)

    cross_covariance_matrix = cross_covariance(P,X, P_nearest_neighbors, up, ux)

    cc_transpose_difference = cross_covariance - cross_covariance.T
    Delta = np.array([cc_transpose_difference[1, 2], cc_transpose_difference[2, 0], cc_transpose_difference[0, 1]])

    trace           = np.trace(cross_covariance_matrix)
    transpose_trace = np.trace(cross_covariance_matrix.T)

    Q = np.zeros((4,4))
    Q[0,0]     = trace
    Q[0,1:4]   = Delta.T
    Q[1:4,0]   = Delta
    Q[1:4,1:4] = cross_covariance_matrix + cross_covariance_matrix.T - trace * np.identity(3)

    w, v = np.linalg.eig(Q)
    max_eigenvalue_index = max_index(w)
    max_eigenvalue  = v[max_eigenvalue_index]
    max_eigenvector = w[max_eigenvalue_index]

    # this max_eigenvector is the rotation quaternion we need,
    # so make it a rotation matrix
    return quaternion_to_rotation_matrix(max_eigenvector)

def optimal_matricies(P, X, P_nearest_neighbors = None, up = None, ux = None):
    return optimal_rotation_matrix(P,X,P_nearest_neighbors,up,ux), translation_matrix(up-ux)

def mean_square_error(P,X):
    error = 0
    for i in range(len(P)):
        error += np.linalg.norm(P[i] - X[i], order=2)

    return error / len(P)

def icp(P,X,up = None, ux = None, threshold = 0.9):
    """Returns the transformation matrix to take the point cloud X to the
    point cloud P by rigid transformation. If up and ux are specified, rotations
    and translations are relative to up on P and ux on X, which remain fixed in the
    transformation."""
    matrix = np.identity(4)
    X_copy = np.copy(X)

    if up == None:
        up = center_of_mass(P)
    if ux == None:
        ux = center_of_mass(X)

    P_nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P)

    while True:
        rot, tr = optimal_matricies(P, X_copy, P_nearest_neighbors, up, ux)
        matrix = dot(matrix, rot)
        matrix = dot(matrix, tr)

        # apply matricies
        dot(rot, X_copy)
        dot(tr,  X_copy)

        if mean_square_error(P, X_copy) < threshold
            break

    return matrix
