"""
geometry.py
Contains a number of utility functions to expediate
3D geometric programming, with points specified as numpy arrays.
"""
import numpy as np
import math
from operator import add

def dist(p, q):
    """Returns the L2 distance between points p and q."""
    return np.linalg.norm((p - q), 2)

def unit_vector(v):
    """Provided a vector v, returns the unit vector pointing
    in the same direction as v."""
    return v / np.linalg.norm(v, 2)

def center_of_mass(cloud):
    """Returns the 'center of mass', or average point,
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

def translation_matrix(dv):
    """Returns a 4x4 affine translation matrix which translates each point by
    the 3x1 vector dv."""
    M = np.identity(4)

    M[0, 3] = dv[0]
    M[1, 3] = dv[1]
    M[2, 3] = dv[2]

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
