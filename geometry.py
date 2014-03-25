import numpy as np
import math
from operator import add

def dist(P,Q):
    """Returns the L2 distance between points P and Q."""
    return np.linalg.norm(P - Q, 2)

def unit_vector(v):
    return v / np.linalg.norm(v, 2)

def center_of_mass(P):
    """Returns the 'center of mass', or average point,
    in the point cloud P."""
    return reduce(add, P) / float(len(P))

def bounding_box(P):
    """Returns two points, opposing corners of the minimum
    bounding box (orthotope) that contains the point cloud P."""
    minx = 0
    maxx = 0
    miny = 0
    maxy = 0
    minz = 0
    maxz = 0
    for p in P:
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

    return np.array([[minx,miny,minz], [maxx,maxy,maxz]])

def estimate_max_diagonal(P):
    """Estimates the longest distance between two points
    in the point set P in O(n), by computing the center of mass,
    then finding the point farthest away from the COM, then finding
    the point farthest away from that point, and returning the distance
    between these two points."""
    com = center_of_mass(P)

    p0 = farthest_from(com, P)
    p1 = farthest_from(p0, P)

    return np.linalg.norm(p1 - p0, 2)

def farthest_from(pt, P):
    """Returns the point in P which is farthest
    from the point pt."""
    furthest_from_pt          = None
    furthest_from_pt_distance = 0
    for p in P:
        p_distance = np.linalg.norm(com - p, 2)
        if p_distance > furthest_from_pt_distance:
            furthest_from_pt = p
            furthest_from_pt_distance = p_distance

    return furthest_from_pt

def single_rotation_matrix(theta, axis = 0):
    M = np.identity(3)

    M[(axis + 1) % 3, (axis + 1) % 3] =  math.cos(theta)
    M[(axis + 2) % 3, (axis + 2) % 3] =  math.cos(theta)
    M[(axis + 2) % 3, (axis + 1) % 3] =  math.sin(theta)
    M[(axis + 1) % 3, (axis + 2) % 3] = -math.sin(theta)

    return M

def rotation_matrix(theta_x, theta_y, theta_z):
    return np.dot(single_rotation_matrix(theta_x, 0), 
             np.dot(single_rotation_matrix(theta_y, 1), single_rotation_matrix(theta_z, 2))
           )

def translation_matrix(dv):
    M = np.identity(4)

    M[0,3] = dv[0]
    M[1,3] = dv[1]
    M[2,3] = dv[2]

    return M

def scaling_matrix(dv):
    M = np.identity(4)

    M[0,0] = dv[0]
    M[1,1] = dv[1]
    M[2,2] = dv[2]

    return M

def promote(M, w = 1):
    """Promotes the 3x3 matrix M to a 4x4 matrix by just adding zeros. Entry
    4,4 will be equal to w (useful for translation matrices)."""
    A = np.zeros((4,4))
    A[0:3,0:3] = M
    A[3,3] = w

    return A

def apply_transform(M, X):
    """Apply the matrix M to all the vectors in X, which
    are row vectors."""
    return np.dot(M,X.T).T



def quaternion_to_rotation_matrix(q):
    """Returns a 3x3 matrix for rotation in R3 corresponding
    to the quaternion vector q=[q0 q1 q2]."""
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    return np.array([
        [q0**2 + q1**2 - q2**2 - q3**2,  2*(q1*q2 - q0*q3),              2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),              q0**2 + q2**2 - q1**2 - q3**2,  2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),              2*(q2*q3 + q0*q1),              q0**2 + q3**2 - q1**2 - q2**2]])
