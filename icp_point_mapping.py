from point_mapping import *
from identify import *
import numpy as np
from geometry import *
import cv2

import random

from operator import add
from objfile import *

class IcpAlgorithm(RegistrationAlgorithm):
    def __init__(self, source_mesh, destination_mesh, 
                 source_fixed = None, destination_fixed = None, max_iterations = 100):
        super(RegistrationAlgorithm, self).__init__(source_mesh, destination_mesh)

        self.source_fixed      = source_fixed
        self.destination_fixed = destination_fixed

        self.A = None

    def run(self):
        """Calls icp(), which takes a long time, to find the registration
        between the two meshes."""
        self.A, self.global_confidence = icp(self.source_mesh,  self.destination_mesh, 
                                             self.source_fixed, self.destination_fixed)

    def transform(self, source_point):
        assert(self.A is not None)
        # No reason to assume this point is better or worse
        # than the whole mapping, so return confidence of 1.0
        return np.dot(self.A, source_point), 1.0

EPSILON = 0.01

CONVERGENCE_THRESHOLD = 1.0e-8

DO_SCALE = True

def scaling_step(P,X):
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
    center of masses of each point cloud, as defined in [Besl1992]."""

    if P_nearest_neighbors == None:
        P_nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P)

    # make an array of pairs (pi,xi) for each xi in X, such that
    # for xi, pi is the closest point to xi in P.
    matching = identify_points(P, X, P_nearest_neighbors)[0]

    if up == None:
        up = center_of_mass(P)
    if ux == None:
        ux = center_of_mass(X)

    return reduce(add, [np.outer(X[i] - ux, p - up) for i,p in enumerate(matching)]) / float(len(P))


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

    # this max_eigenvector is the rotation quaternion we need,
    # so make it a rotation matrix
    return promote(quaternion_to_rotation_matrix(unit_vector(max_eigenvector)))

def optimal_matricies(P, X, P_nearest_neighbors = None, up = None, ux = None):
    rot = optimal_rotation_matrix(P,X,P_nearest_neighbors,up,ux)
    return rot, translation_matrix(up - np.dot(rot, ux))

def mean_square_error(P,X):
    error = 0
    for i in range(min([len(P), len(X)])):
        error += dist(P[i] - X[i]) # L2 norm, see geometry.py

    return error / len(P)


DO_SHAKE = True
SHAKE_AMOUNT = 1.5
SHAKE_THRESHOLD = 1e-6

VERBOSE = False

def get_shake_matrix(rotate_amount = SHAKE_AMOUNT, translate_amount = SHAKE_AMOUNT):
    """Returns an affine matrix, comprised of a random
    rotation and translation, that can be used to try and
    shake the icp out of a local minima that is not ideal."""
    shake = promote(rotation_matrix(random.uniform(-shake_amount, shake_amount),
                                    random.uniform(-shake_amount, shake_amount),
                                    random.uniform(-shake_amount, shake_amount)))
    shake = np.dot(shake, translation_matrix(np.array([random.uniform(-translate_amount, translate_amount),
                                                       random.uniform(-translate_amount, translate_amount),
                                                       random.uniform(-translate_amount, translate_amount)])))
    return shake

def icp(P,X,up = None, ux = None, P_nearest_neighbors = None, max_iterations = 100):
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

    if P_nearest_neighbors == None:
        P_nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P)

    last_error   = mean_square_error(P,X_copy)
    lowest_error = last_error

    global_matrix = scaling_step(P,X)

    X_copy = apply_transform(global_matrix, X_copy)
    ux     = apply_transform(global_matrix, ux)

    best_global_matrix = global_matrix

    for x in range(max_iterations):
        # if X has more points than P,
        # randomly subsample from X, differently
        # each iteration.
        if len(X_copy) > len(P):
            X_sample = np.array(random.sample(X_copy, len(P)))
        else:
            X_sample = X_copy

        rot, tr = optimal_matricies(P, X_sample, P_nearest_neighbors, up, ux)

        # for column vectors on right, rotate, then translate
        matrix = np.dot(tr, rot)

        global_matrix = np.dot(matrix, global_matrix)

        X_copy = apply_transform(matrix, X_copy)
        ux = apply_transform(matrix, ux)

        # ensures that ux remains close to up
        # assert(np.allclose(center_of_mass(X_copy), up))

        if last_error < CONVERGENCE_THRESHOLD:
            break


        current_error = mean_square_error(P, X_copy)

        # error should decrease with every step
        if (last_error < current_error or abs(last_error - current_error) < SHAKE_THRESHOLD) and DO_SHAKE:
            if VERBOSE:
                print "Error has not improved. Shaking."
            
            shake = get_shake_matrix()

            global_matrix = np.dot(shake, global_matrix)
            X_copy = apply_transform(shake, X_copy)
            ux = apply_transform(shake, ux)

        if VERBOSE:
            print "Iteration {0:>3}: Last Error: {1:f} Lowest Error {2:f}".format(
                    x, last_error, lowest_error
                    )

        if last_error < lowest_error:
            lowest_error = last_error
            best_global_matrix = global_matrix

        last_error    = current_error


    print "Lowest Mean Squared Error:", lowest_error
    return best_global_matrix, lowest_error

def identify_points(P, X, P_nearest_neighbors = None):
    """Returns a point cloud X' by matching each point on X
    with its closest neighbor on P, as well as an array of 
    relative confidences on the interval (0,1), determined
    by the distance each point must move to get to its mapping."""

    if P_nearest_neighbors == None:
        P_nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(P)

    distances, indices = P_nearest_neighbors.kneighbors(X)

    max_dist = max(distances)
    scaled_distances = [d / max_dist for d in distances]
    # make an array of pairs (pi,xi) for each xi in X, such that
    # for xi, pi is the closest point to xi in P.
    matching = []
    for i in range(len(indices)):
        matching.append(P[indices[i][0]])

    return np.array(matching), scaled_distances
