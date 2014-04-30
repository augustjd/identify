"""
curvature_point_mapping.py

Defines the CurvatureAlgorithm class which performs an algorithm defined
below to identify two point clouds. This algorithm is identical to
RadialAlgorithm except that the fitting being evaluated, Efit, is a fitting
between high-curvature points on each mesh, not the meshes themselves.

Algorithm: 

    1. Perform the scaling step as before.  
    2. Identify the grasping points trivially by a translation matrix, and
    apply this matrix to the source mesh.

    3. Determine the points farthest away from the grasping point on each
    mesh, and, using get_unify_segments_matrix() from geometry.py, rotate the
    source mesh about the grasp point until the line joining the farthest
    point to the grasp point is the same on both meshes. This defines an
    axis of rotation and reduces the problem to a single degree of freedom,
    the theta of rotation about this axis.

    4. Create a range of k guesses for this theta, 2pi/i for i in range(1,k),
    and run seperate optimization procedures for each guess, which attempt
    to refine the guess to a local minimum to Efit, defined similarly
    to in the other cases.

    5. Report the guess whose convergence results in the smallest Efit as the
    estimate for theta, and transform the source_mesh by this rotation,
    returning the total transformation up to this point as the result.
"""
from point_mapping import *
import numpy as np
import scipy
from geometry import *

import random

from operator import add
from objfile import *
import trimesh

from icp_point_mapping import *
from radial_point_mapping import RadialAlgorithm

VERBOSE = False
MAX_SAMPLE_SIZE = 10

class CurvatureAlgorithm(RadialAlgorithm):
    def __init__(self, source_mesh, destination_mesh, 
                 source_fixed = None, destination_fixed = None, 
                 max_iterations = 10):
        super(RadialAlgorithm, self).__init__(source_mesh,
                destination_mesh, source_fixed, destination_fixed)

        self.iteration = 1
        self.max_iterations = max_iterations

        self.origin      = destination_fixed

        self.transformations = []

        self.min_energy = float('+inf')

        self.source_high_curvature_indices = get_indices_of_high_curvature(self.source_mesh)
        self.source_high_curvature_indices.sort()

        self.destination_high_curvature = map(self.destination_mesh.vs.__getitem__,
                get_indices_of_high_curvature(self.destination_mesh))

        self.destination_nearest_neighbors = NearestNeighbors(n_neighbors=1, 
                algorithm="kd_tree").fit(self.destination_high_curvature)

        self.verbose = True

        from threading import Lock
        self.lock = Lock()

    MAX_SAMPLE_SIZE = 500

    def Efit(self, transformed_points):
        sample = map(transformed_points.__getitem__, self.source_high_curvature_indices)
        return 100*sum(nearest_neighbor_distance(pt, self.destination_nearest_neighbors)**2 for pt in sample)

    def rotate_and_get_energy(self, theta):
        A,b = from_affine(self.fixed_axis_rotation_matrix(theta))
        copy = (np.dot(A, self.source_as_mat.T).T + b)
        energy = self.Efit(copy)
        with self.lock:
            if energy < self.min_energy:
                if self.verbose:
                    print "New best energy:", energy
                self.min_energy = energy
                self.vertices = np.asarray(copy)
        return energy

class CurvatureIcp(IcpAlgorithm):
    def run(self):
        """Calls icp(), which takes a long time, to find the registration
        between the two meshes."""
        #self.source_fixed      = center_of_mass(self.source_mesh.vs)
        #self.destination_fixed = center_of_mass(self.destination_mesh.vs)

        source_high_curvature      = get_mesh_of_high_curvature(self.source_mesh)
        destination_high_curvature = get_mesh_of_high_curvature(self.destination_mesh)

        self.matrix, error = icp(source_high_curvature.vs,  
                                 destination_high_curvature.vs, 
                                 self.source_fixed, 
                                 self.destination_fixed,
                                 self.max_iterations)

        self.inverse = np.linalg.inv(self.matrix)
        self.global_confidence = 1 - error
        print self.global_confidence
