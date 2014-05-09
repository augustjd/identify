"""
radial_point_mapping.py

Defines the RadialAlgorithm class which performs an algorithm defined below to
identify two point clouds. Unlike the other FixedPairRegistrationAlgorithms, 
this one REQUIRES defined grasping points which are distinct from the center of mass.

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
import scipy.optimize
from geometry import *

import random

from operator import add
from objfile import *
import trimesh

from icp_point_mapping import scaling_step

from multithreading_help import *

MAX_SAMPLE_SIZE = 10 
class RadialAlgorithm(FixedPairRegistrationAlgorithm):
    def __init__(self, source_mesh, destination_mesh, 
                 source_fixed = None, destination_fixed = None, 
                 max_iterations = 10, verbose = False):
        super(RadialAlgorithm, self).__init__(source_mesh,
                destination_mesh, source_fixed, destination_fixed, verbose =
                verbose)
        self.destination_nearest_neighbors = NearestNeighbors(n_neighbors=1, 
                algorithm="kd_tree").fit(destination_mesh.vs)

        self.iteration = 1
        self.max_iterations = max_iterations

        self.origin      = destination_fixed

        self.transformations = []

        self.min_energy = float('+inf')

    MAX_SAMPLE_SIZE = 500
    def Efit(self, transformed_points):
        sample = random.sample(transformed_points, 
                               min(len(transformed_points), MAX_SAMPLE_SIZE)
                 )
        return 100*sum(nearest_neighbor_distance(pt, self.destination_nearest_neighbors)**2 for pt in sample)

    def scaling_step(self, source_as_mat):
        scale = np.asmatrix(scaling_step(self.destination_mesh.vs, self.source_mesh.vs)[0:3,0:3])

        # scaling involes no translation, so ignore second component
        scale_A = scale

        source_as_mat *= scale_A
        self.source_fixed = np.dot(scale_A, self.source_fixed)

        self.transformations.append(promote(scale))

    def identify_grasp_points(self, source_as_mat):
        grasp_point_difference = self.destination_fixed - self.source_fixed

        source_as_mat += grasp_point_difference

        self.transformations.append(translation_matrix(grasp_point_difference))
        return grasp_point_difference

    def unify_grasp_axes(self, source_as_mat):
        source_most_distant = farthest_from(
                self.destination_fixed, np.asarray(source_as_mat))

        destination_most_distant = farthest_from(
                self.destination_fixed, self.destination_mesh.vs)
        
        if self.verbose:
            print source_most_distant, destination_most_distant

        self.axis = self.destination_fixed - destination_most_distant
        affine = np.asmatrix(get_unify_segments_matrix(destination_most_distant,
            self.destination_fixed, source_most_distant))

        source_most_distant = apply_transform(affine, source_most_distant, 3)
        self.source_axis = self.destination_fixed - source_most_distant

        self.transformations.append(affine)

        return affine

    def rotate_and_get_energy(self, theta):
        A,b = from_affine(self.fixed_axis_rotation_matrix(theta))
        copy = np.dot(A, self.source_as_mat.T).T + b
        energy = self.Efit(copy)
        if energy < self.min_energy:
            if self.verbose:
                print "New best energy:", energy
            self.min_energy = energy
            self.vertices = np.asarray(copy)
        return energy

    def make_guess(self, guess):
        import sys
        if self.verbose:
            sys.stdout.write("Evaluating theta = {0}...".format(guess))
            sys.stdout.flush()

        result = scipy.optimize.minimize(self.rotate_and_get_energy, guess,
                options = { 'maxiter': 3 })

        if self.verbose:
            print "Final energy: {0}".format(self.rotate_and_get_energy(guess))

        return result

    def make_guesses(self, source_as_mat, num_guesses):
        guesses = [k * 2 * math.pi / num_guesses for k in range(0, num_guesses)]
        self.source_as_mat = source_as_mat

        results = map(self.make_guess, guesses)

        errors  = [self.rotate_and_get_energy(result.x) for result in results]

        return results, errors

    def fixed_axis_rotation_matrix(self, theta):
        return arbitrary_axis_rotation_at_arbitrary_origin(self.axis,
                self.origin, theta)

    def run(self, num_guesses = 30):
        import math
        self.transformations = []

        source_as_mat = np.asmatrix(trimesh.asarray(self.source_mesh.vs))
        # step 1
        self.scaling_step(source_as_mat)

        # step 2
        dv = self.identify_grasp_points(source_as_mat)
        self.source_fixed += dv
        assert(np.allclose(self.source_fixed, self.destination_fixed))

        # step 3
        M = self.unify_grasp_axes(source_as_mat)
        source_as_mat = apply_transform(np.asarray(M), np.asarray(source_as_mat), 3)
        assert(np.allclose(self.source_fixed, self.destination_fixed))

        # step 4
        results, errors = self.make_guesses(source_as_mat, num_guesses)
        
        # for now, just pick single result which produces lowest error 
        def choose_result(results, errors):
            best_index = errors.index(min(errors))
            return results[best_index].x, errors[best_index]

        self.theta_estimate, self.error = choose_result(results, errors)
        if self.verbose:
            print "Best theta estimate: {0} Error: {1}".format(self.theta_estimate, self.error)
        final_rotation = self.fixed_axis_rotation_matrix(self.theta_estimate)

        self.transformations.append(final_rotation)

        print "\n".join(map(str, self.transformations))

        self.A, self.b = from_affine(compose(*reversed(self.transformations)))
        print self.A
        print self.b

    def transform(self, source_point):
        # TODO utilize the error above to provide more info to caller
        return (np.dot(self.A, source_point) + self.b, 1.0)

    def transformed_mesh(self):
        copy = self.source_mesh.copy()
        copy.vs = np.asarray(self.vertices)
        copy.positions_changed()

        return copy
