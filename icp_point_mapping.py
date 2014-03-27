"""
icp_point_mapping.py

Defines the IcpAlgorithm class which inherits from RegistrationAlgorithm,
which implements the ICP (iterative closest point) algorithm for registration
between two point clouds.
"""

from point_mapping import *
import numpy as np
from geometry import *
import cv2

import random

from operator import add
from objfile import *

class IcpAlgorithm(RegistrationAlgorithm):
    def __init__(self, source_mesh, destination_mesh, 
                 source_fixed_index = None, destination_fixed_index = None, 
                 max_iterations = 100):
        super(IcpAlgorithm, self).__init__(source_mesh, destination_mesh)

        if source_fixed_index is None:
            self.source_fixed = center_of_mass(source_mesh.vs)
        else:
            self.source_fixed = source_mesh.vs[source_fixed_index]

        if destination_fixed_index is None: 
            self.destination_fixed = center_of_mass(destination_mesh.vs)
        else:
            self.destination_fixed = destination_mesh.vs[destination_fixed_index]

        print "Identifying points Source: {0} Destination: {1}".format(
                    self.source_fixed, self.destination_fixed
                )

        self.matrix = None

        self.max_iterations = max_iterations

    def run(self):
        """Calls icp(), which takes a long time, to find the registration
        between the two meshes."""
        self.matrix, error = icp(self.source_mesh.vs,  
                                 self.destination_mesh.vs, 
                                 self.source_fixed, 
                                 self.destination_fixed,
                                 self.max_iterations)

        self.inverse = np.linalg.inv(self.matrix)
        self.global_confidence = 1 - error
        print self.global_confidence

    def transform(self, source_point):
        assert(self.inverse is not None)
        # No reason to assume this point is better or worse # than the whole mapping, so return confidence of 1.0
        return apply_transform(self.inverse, source_point, len(source_point)), 1.0

EPSILON = 0.01

CONVERGENCE_THRESHOLD = 1.0e-8

DO_SCALE = True

def scaling_step(P, X):
    """Returns a matrix M which scales X to best match the dimensions of P,
    by taking the length of the diagonal of their bounding boxes."""
    P_diagonal_length = estimate_max_diagonal(P)
    X_diagonal_length = estimate_max_diagonal(X)

    if not DO_SCALE:
        scale_factor = 1.0
    else:
        scale_factor = P_diagonal_length / X_diagonal_length

    print "Scaling X by factor of " + str(scale_factor)

    return scaling_matrix(np.array([scale_factor, scale_factor, scale_factor]))

def index_of_max(arr):
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
    center of masses of each point cloud, as defined in [Besl1992]."""

    if P_nearest_neighbors == None:
        P_nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P)

    # make an array of pairs (pi, xi) for each xi in X, such that
    # for xi, pi is the closest point to xi in P.
    matching = identify_points(P, X, P_nearest_neighbors)[0]

    if up == None:
        up = center_of_mass(P)
    if ux == None:
        ux = center_of_mass(X)

    return reduce(add, [np.outer(X[i] - ux, p - up) for i, p in enumerate(matching)]) / float(len(P))


def optimal_rotation_matrix(P, X, P_nearest_neighbors = None, up = None, ux = None):
    """Returns the optimal rotation matrix Q of the point set X for minimizing
    its covariance with P, a 3x3 matrix. See [Besl92] for details."""
    if P_nearest_neighbors == None:
        P_nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P)

    cross_covariance_matrix = cross_covariance(P, X, P_nearest_neighbors, up, ux)

    cc_transpose_difference = cross_covariance_matrix - cross_covariance_matrix.T
    Delta = np.array([cc_transpose_difference[1, 2], cc_transpose_difference[2, 0], cc_transpose_difference[0, 1]])

    trace           = np.trace(cross_covariance_matrix)
    transpose_trace = np.trace(cross_covariance_matrix.T)

    Q = np.zeros((4, 4))
    Q[0, 0]     = trace
    Q[0, 1:4]   = Delta.T
    Q[1:4, 0]   = Delta
    Q[1:4, 1:4] = cross_covariance_matrix[0:3, 0:3] + cross_covariance_matrix.T[0:3, 0:3] - trace * np.identity(3)

    v, w = np.linalg.eig(Q)
    max_eigenvalue_index = index_of_max(v)
    # max_eigenvalue  = v[max_eigenvalue_index]
    max_eigenvector = w[:, max_eigenvalue_index]

    # this max_eigenvector is the rotation quaternion we need,
    # so make it a rotation matrix
    return promote(quaternion_to_rotation_matrix(unit_vector(max_eigenvector)))

def optimal_matrices(P, X, P_nearest_neighbors = None, up = None, ux = None):
    """Returns the optimal rotation and translation matricies for
    rigidly transforming X to match P, and also identifies points up and ux."""
    rot = optimal_rotation_matrix(P, X, P_nearest_neighbors, up, ux)
    tr  = translation_matrix(up - apply_transform(rot, ux, len(ux)))

    assert(np.allclose(up, apply_transform(tr, apply_transform(rot, ux, len(ux)), len(ux))))

    return rot, tr

def mean_square_error(P, X):
    """Returns the sum of the L2 norms between closest points on P and X,
    measuring the difference between the two point clouds."""
    error = 0.0
    for i in range(min([len(P), len(X)])):
        error += dist(P[i], X[i]) # L2 norm, see geometry.py

    return error / len(P)


DO_SHAKE = True
SHAKE_AMOUNT = 5.5
SHAKE_THRESHOLD = 1e-6

VERBOSE = True

def get_shake_matrix(rotate_amt = SHAKE_AMOUNT, translate_amt = SHAKE_AMOUNT):
    """Returns an affine matrix, comprised of a random rotation and
    translation, that can be used to try and shake the icp out of a local
    minima that is not ideal."""
    def rand(amt):
        return random.uniform(-amt, amt)

    return compose( promote(rotation_matrix(rand(rotate_amt), 
                                            rand(rotate_amt),
                                            rand(rotate_amt))),
                    translation_matrix(np.array([rand(translate_amt),
                                                 rand(translate_amt),
                                                 rand(translate_amt)])) )

class IcpState:
    """Groups together data and simple functions to allow a cleaner icp()
    function."""
    def __init__(self, P, X, ux, up):
        self.global_matrix = np.identity(4)
        self.P             = P
        self.X_copy        = np.copy(X)

        self.ux            = ux.copy()
        self.up            = up

    def apply_transform_to_all(self, transform):
        self.global_matrix = np.dot(transform, self.global_matrix)
        self.X_copy        = apply_transform(transform, self.X_copy, 3)
        self.ux            = apply_transform(transform, self.ux, 3)

    def sample(self):
        if len(self.X_copy) > len(self.P):
            return np.array(random.sample(self.X_copy, len(self.P)))
        else:
            return self.X_copy


    def error(self):
        return mean_square_error(self.P, self.X_copy)

def icp(P, X, up = None, ux = None, max_iterations = 100, P_nearest_neighbors = None):
    """Returns the transformation matrix to take the point cloud X to the
    point cloud P by rigid transformation. If up and ux are specified, rotations
    and translations are relative to up on P and ux on X, which remain fixed in the
    transformation."""
    if up == None:
        up = center_of_mass(P)
    if ux == None:
        ux = center_of_mass(X)

    if P_nearest_neighbors == None:
        P_nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P)

    state = IcpState(P, X, ux, up)

    state.apply_transform_to_all(scaling_step(P, X))

    best_global_matrix = state.global_matrix

    last_error   = state.error()
    lowest_error = float('+inf')

    for x in range(max_iterations):
        # if X has more points than P,
        # randomly subsample from X, differently
        # each iteration.
        X_sample = state.sample()

        rot, tr = optimal_matrices(P, X_sample, P_nearest_neighbors, state.up, state.ux)

        # for column vectors on right, rotate, then translate
        matrix = np.dot(tr, rot)

        state.apply_transform_to_all(matrix)

        # ensures that ux remains close to up
        #print "up {0} ux {1}".format(state.up, state.ux)
        #print "COM P: {0} COM X: {1}".format(center_of_mass(P), center_of_mass(state.X_copy))
        assert(np.allclose(state.ux, state.up, 1e-6))

        current_error = state.error()

        # error should decrease with every step, otherwise shake
        if DO_SHAKE and (last_error < current_error or abs(last_error - current_error) < SHAKE_THRESHOLD):
            if VERBOSE:
                print "Error has not improved. Shaking."
            
            shake = get_shake_matrix()
            state.apply_transform_to_all(shake)

        if VERBOSE:
            print "Iteration {0:>3}: Last Error: {1:f} Lowest Error {2:f} up->ux distance {3}".format(
                    x, last_error, lowest_error, dist(state.up, state.ux)
                    )

        if current_error < lowest_error:
            lowest_error = current_error
            best_global_matrix = state.global_matrix

        last_error    = current_error

        if last_error < CONVERGENCE_THRESHOLD:
            break

    print "Lowest Mean Squared Error:", lowest_error
    return best_global_matrix, lowest_error

def identify_points(P, X, P_nearest_neighbors = None):
    """Returns a point cloud X' by matching each point on X
    with its closest neighbor on P, as well as an array of 
    relative confidences on the interval (0, 1), determined
    by the distance each point must move to get to its mapping."""

    if P_nearest_neighbors == None:
        P_nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P)

    distances, indices = P_nearest_neighbors.kneighbors(X)

    max_dist = max(distances)

    if max_dist == 0:
        max_dist = 1e-6

    scaled_distances = [d / max_dist for d in distances]
    # make an array of pairs (pi, xi) for each xi in X, such that
    # for xi, pi is the closest point to xi in P.
    matching = []
    for i in range(len(indices)):
        matching.append(P[indices[i][0]])

    return np.array(matching), scaled_distances
