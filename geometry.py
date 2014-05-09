"""
geometry.py
Contains a number of utility functions to expediate
3D geometric programming, with points specified as numpy arrays.
"""
import numpy as np
import math
from operator import add
from multithreading_help import *

def dist(p, q):
    """Returns the L2 distance between points p and q."""
    return np.linalg.norm((p - q), 2)

def unit_vector(v):
    """Provided a vector v, returns the unit vector pointing
    in the same direction as v."""
    return v / np.linalg.norm(v, 2)

def center_of_mass(cloud):
    """Returns the center of mass, or average point,
    in the point cloud 'cloud'."""
    return reduce(add, cloud) / float(len(cloud))

def bounding_box(cloud):
    """Returns two points, opposing corners of the minimum
    bounding box (orthotope) that contains the point cloud 'cloud'."""
    minx = 0
    maxx = 0
    miny = 0
    maxy = 0
    minz = 0
    maxz = 0
    for p in cloud:
        if p[0] < minx:
            minx = p[0]
        if p[0] > maxx:
            maxx = p[0]

        if p[0] < miny:
            miny = p[0]
        if p[0] > maxy:
            maxy = p[0]

        if p[0] < minz:
            minz = p[0]
        if p[0] > maxz:
            maxz = p[0]

    return np.array([[minx, miny, minz], [maxx, maxy, maxz]])

def estimate_max_diagonal(cloud):
    """Estimates the longest distance between two points in the point cloud
    'cloud' in O(n), by computing the center of mass, then finding the point
    farthest away from the COM, then finding the point farthest away from that
    point, and returning the distance between these two points."""
    com = center_of_mass(cloud)

    p0 = farthest_from(com, cloud)
    p1 = farthest_from(p0, cloud)

    return np.linalg.norm(p1 - p0, 2)

def farthest_from(point, cloud):
    """Returns the point in cloud which is farthest from the point 'point'."""
    furthest_from_pt          = None
    furthest_from_pt_distance = 0
    for p in cloud:
        p_distance = np.linalg.norm(point - p, 2)
        if p_distance > furthest_from_pt_distance:
            furthest_from_pt = p
            furthest_from_pt_distance = p_distance

    return furthest_from_pt

def single_rotation_matrix(theta, axis = 0):
    """Returns a 3x3 rotation matrix of theta radians about the provided axis;
    0 = x, 1 = y, 2 = z."""
    M = np.identity(3)

    M[(axis + 1) % 3, (axis + 1) % 3] =  math.cos(theta)
    M[(axis + 2) % 3, (axis + 2) % 3] =  math.cos(theta)
    M[(axis + 2) % 3, (axis + 1) % 3] =  math.sin(theta)
    M[(axis + 1) % 3, (axis + 2) % 3] = -math.sin(theta)

    return M

def rotation_matrix(theta_x, theta_y, theta_z):
    """Returns a 3x3 rotation matrix which rotates theta_x about the x-axis,
    then theta_y about the y-axis, and then theta_z about the z-axis."""
    return compose(single_rotation_matrix(theta_x, 0), 
                   single_rotation_matrix(theta_y, 1), 
                   single_rotation_matrix(theta_z, 2))

def arbitrary_axis_rotation(axis, theta):
    """Returns a 4x4 rotation matrix which rotates input vectors by theta
    radians about the provided axis, using a right-handed coordinate system.
    axis should be a numpy.ndarray., theta is a float.  Uses the derivation
    found at http://science.kennesaw.edu/~plaval/math4490/rotgen.pdf"""
    import math

    assert(not np.allclose(axis, np.zeros(3)))

    x, y, z = unit_vector(axis)
    C = math.cos(theta)
    S = math.sin(theta)
    t = 1.0 - C

    result = np.array([[t*x**2 + C,  t*x*y - S*z, t*x*z + S*y,  0],
                       [t*x*y + S*z, t*y**2 + C,  t*y*z - S*x,  0],
                       [t*x*z - S*y, t*y*z + S*x, t*z**2 + C,   0],
                       [0,           0,           0,            1]])

    assert(abs(np.linalg.det(result)) - 1.0 < 1e-3)

    return result

def arbitrary_axis_rotation_at_arbitrary_origin(axis, origin, theta):
    """Returns a 4x4 affine transformation which rotates input vectors
    by theta radians about the provided axis, centered at the provided
    origin, using a right-handed coordinate system."""
    return compose(translation_matrix(origin), 
                   arbitrary_axis_rotation(axis, theta), 
                   translation_matrix(-origin))


def unitary_matrix(M):
    # must be square
    assert(M.shape[0] == M.shape[1])

    return M / (np.linalg.det(M) ** (1./M.shape[0]))

def get_unify_segments_matrix(a, b, c):
    """Returns a matrix which rotates the line segment bc to put it
    on the line defined by segment ab."""
    import math

    # get first 3 components, if these are affine 4x1 vectors,
    # so that cross is well defined.
    a = vector_from_affine(a)
    b = vector_from_affine(b)
    c = vector_from_affine(c)

    # unit_vector(cross) fails if they are identical,
    # so in that case return the identity.
    if np.allclose(np.array([0,0,0]), (a-b) - (c-b)):
        return np.identity(4)

    axis   = unit_vector(np.cross(c-b, a-b))
    origin = b
    theta  = math.acos(min(1.0, np.dot(unit_vector(a-b), unit_vector(c-b))))

    return unitary_matrix(arbitrary_axis_rotation_at_arbitrary_origin(axis,
        origin, theta))

def vector_to_affine(v):
    result = np.ones(4)
    result[0:3] = v.copy()
    return result

def vector_from_affine(v):
    return v[0:3].copy()

def to_affine(A,b):
    """Turns a 3x3 ndarray and 3x1 ndarray pair (A,b) into an
    equivalent affine 4x4 matrix."""
    result = np.identity(4)
    result[0:3, 0:3] = A
    result[0:3, 3]   = b
    return result

def from_affine(affine):
    """Given an affine transformation T, return the pair (A,b) such that the
    action Tx on a 4d vector x is equivalent to the action Ax' + b on the 3d vector
    x' produced by ignoring x's w component."""
    return (affine[0:3, 0:3], affine[0:3, 3])

def translation_matrix(dv):
    """Returns a 4x4 affine translation matrix which translates each point by
    the 3x1 vector dv."""
    M = np.identity(4)
    M[0:3, 3] = dv[0:3]

    return M

def scaling_matrix(dv):
    """Returns a 4x4 affine scaling matrix which stretches the x axis by
    dv[0], y axis by dv[1], and z axis by dv[2]."""
    M = np.identity(4)

    M[0, 0] = dv[0]
    M[1, 1] = dv[1]
    M[2, 2] = dv[2]

    return M

def promote(M, w = 1):
    """Promotes the 3x3 matrix M to a 4x4 matrix by just adding zeros. Entry
    4,4 will be equal to w (useful for translation matrices)."""
    A = np.zeros((4, 4))
    A[0:3, 0:3] = M
    A[3, 3] = w

    return A

def apply_transform(M, X, cols = 4):
    """Apply the matrix M to all the vectors in X, which are row vectors.
    Conveniently, it will apply M even if M is affine and X is comprised
    of 3x1 vectors, using decompose_affine()."""
    if   cols == 3:
        A, b = decompose_affine(M)
        return np.dot(A, X.T).T + b
    elif cols == 4:
        return np.dot(M, X.T).T

def quaternion_to_rotation_matrix(q):
    """Returns a 3x3 matrix for rotation in R3 corresponding to the quaternion
    vector q=[q0 q1 q2]."""
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    return np.array([
        [q0**2 + q1**2 - q2**2 - q3**2,  2*(q1*q2 - q0*q3),              2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),              q0**2 + q2**2 - q1**2 - q3**2,  2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),              2*(q2*q3 + q0*q1),              q0**2 + q3**2 - q1**2 - q2**2]])

def compose(*args):
    """Composes input matrices into a single matrix by multiplying them
    together from left to right."""
    return reduce(np.dot, args)

def decompose_affine(M):
    """Returns a tuple, 3x3 rotation matrix A and 3x1 vector b, such that
    Ax+b for 3x1 vector x will result in the same vector as M*[x0 x1 x2 1]."""
    return M[0:3, 0:3].copy(), M[0:3, 3].copy()

def affine_transform_trimesh(mesh, M):
    return transform_trimesh(mesh, lambda p: apply_transform(M, p, 3))

def transform_trimesh(mesh, func):
    for i, vertex in enumerate(mesh.vs):
        mesh.vs[i] = func(vertex)

    mesh.positions_changed()

def mean_square_error(P, X):
    """Returns the sum of the L2 norms between equal-indiced points on P and
    X, approximating the difference between the two point clouds.""" 
    error = 0.0
    for i in range(min(len(P), len(X))):
        error += dist(P[i], X[i]) # L2 norm, see geometry.py

    return error / len(P)

def nearest_neighbor_sampling_error(P, X, P_nearest_neighbors, sample_size = 1000):
    """Returns the sum of the L2 norms between a subset of closest points on P and X,
    estimating the difference between the two point clouds.""" 
    import random
    sample_size = min(len(X), sample_size)
    sample      = np.array(random.sample(X, sample_size))

    distances, indices = P_nearest_neighbors.kneighbors(sample)

    return reduce(lambda arr, y: arr[0] + y, distances)[0] / sample_size

def nearest_neighbor_distance(point, P_nearest_neighbors):
    return P_nearest_neighbors.kneighbors(np.array(point))[0][0][0]

def nearest_neighbor_index(point, P_nearest_neighbors):
    return P_nearest_neighbors.kneighbors(np.array(point))[1][0][0]

def estimate_grasp_point(vs, sample_size = 5):
    """Estimates the grasp point on mesh by taking an average of the topmost
    (highest Y) sample_size points on mesh."""
    actual_sample_size = min(len(vs), sample_size)
    return center_of_mass(sorted(vs, key=lambda v: v[1])[-actual_sample_size:])

def curvature_of_edge(he, mesh):
    """Computes the curvature at the provided halfedge of the TriMesh mesh."""
    edge_indices = mesh.edges[he.edge]

    if edge_indices[1] == he.to_vertex:
        # flip the halfedge
        he = mesh.halfedges[he.opposite_he]

    return he.get_curvature(mesh)

def curvature_at_point(tup):
    i, mesh = tup
    """Returns the curvature at the point at index i on mesh.
    NOTE: If the point is on a boundary, returns +inf."""
    if mesh.vertex_is_boundary(i):
        return i, float("+inf")

    vertex_neighbors = mesh.vertex_vertex_neighbors(i)
    # some halfedges are directed outward, some inward. we want all of them.
    he_neighbors = []
    for vi in vertex_neighbors:
        outgoing = mesh.directed_edge2he_index((i, vi))
        ingoing  = mesh.directed_edge2he_index((vi, i))

        if outgoing:
            he_neighbors.append(mesh.halfedges[outgoing])
        if ingoing:
            he_neighbors.append(mesh.halfedges[ingoing])

    return i, np.linalg.norm(sum(map(lambda he: curvature_of_edge(he, mesh), he_neighbors))) / 2.0


def get_indices_of_high_curvature(mesh, cutoff_percentile = 0.15):
    """Returns an array of indices of the points on mesh whose absolute value
    of mean curvature is above the cutoff_percentile, by default, this means
    its curvature is higher than 15% of points on the mesh."""
    work = [(i, mesh) for i in range(len(mesh.vs))]
    curvature_of_points = map(curvature_at_point, work)
    curvature_of_points.sort(key=lambda i: i[1])

    cutoff_index = int(math.floor((1 - cutoff_percentile) * len(curvature_of_points)))

    return map(lambda tup: tup[0], curvature_of_points[:cutoff_index])

def get_mesh_of_high_curvature(mesh, cutoff_percentile = 0.15):
    """Returns a copy of mesh which consists only of points whose absolute
    value of mean curvature is above the cutoff_percentile, by default, this
    means its curvature is higher than 15% of points on the mesh."""
    from multiprocessing import Pool, cpu_count
    copy = mesh.copy()

    work = [(i, mesh) for i in range(len(mesh.vs))]
    #curvature_of_points = Pool(processes=cpu_count()).map(curvature_at_point, work)
    curvature_of_points = map(curvature_at_point, work)
    curvature_of_points.sort(key=lambda i: i[1])

    cutoff_index = int(math.floor((1 - cutoff_percentile) * len(curvature_of_points)))
    copy.remove_vertex_indices(map(lambda tup: tup[0], curvature_of_points[:cutoff_index]))

    return copy
