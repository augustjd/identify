"""
li_point_mapping.py

Defines the LiAlgorithm class which inherits from RegistrationAlgorithm, which
implements Hao Li et al's energy minimization algorithm for nonrigid
registration between two point clouds.
"""
from point_mapping import *
import numpy as np
from scipy.optimize import minimize
from geometry import *
import cv2

import random

from operator import add
from objfile import *
import trimesh

from functools import wraps
from time import time

class LiAlgorithm(FixedPairRegistrationAlgorithm):
    def __init__(self, source_mesh, destination_mesh, 
                 source_fixed_point = None, destination_fixed_point = None, 
                 max_iterations = 100):
        super(LiAlgorithm, self).__init__(source_mesh,
                destination_mesh, source_fixed_point, destination_fixed_point)

        self.source_nearest_neighbors = NearestNeighbors(n_neighbors=1, 
                algorithm="kd_tree").fit(source_mesh.vs)
        self.destination_nearest_neighbors = NearestNeighbors(n_neighbors=1, 
                algorithm="kd_tree").fit(destination_mesh.vs)

        # get the indices corresponding to the closest
        # points to the grasp points on each mesh, allowing
        # the correct transformation matrices to be applied,
        # and enforcing the identification of these points
        if self.source_fixed is not None:
            self.source_fixed_index = (
                    nearest_neighbor_index(self.source_fixed,
                                           self.source_nearest_neighbors))
        else:
            self.source_fixed_index = None

        self.source_center_of_mass = center_of_mass(self.source_mesh.vs)

        self.alpha_fit    = 1000
        self.alpha_conf   = 1
        self.alpha_rigid  = 1
        self.alpha_smooth = 0.1

        self.alpha_grasp  = 1000

        self.last_energy = float("inf")

    def energy(self, arr):
        global_rot, global_tr, A_matrices, b_vectors, confidences = self.unpack_flat_array(arr)

        transformed_points = np.array([LiAlgorithm.transform_vertex(self.source_mesh.vs[i], global_rot,
            global_tr, self.source_center_of_mass, A_matrices[i], b_vectors[i]) for i
            in range(len(self.source_mesh.vs))])

        conf_energy = self.alpha_conf * self.Econf(confidences)
        rigid_energy = self.alpha_rigid * self.Erigid(A_matrices)
        grasp_energy = self.alpha_grasp * self.Egrasp(global_rot, global_tr, A_matrices, b_vectors, confidences)        
        smooth_energy = self.alpha_smooth * self.Esmooth(A_matrices, b_vectors, confidences)
        fit_energy = self.alpha_fit * self.Efiteasy(transformed_points)

        energy = fit_energy #+ conf_energy + rigid_energy + grasp_energy + smooth_energy

        print "Efit: {} Econf: {} Erigid: {} Egrasp: {} Esmooth: {}".format(
                fit_energy, conf_energy, rigid_energy, grasp_energy, smooth_energy)

        if abs(energy - self.last_energy) < 1e-5 * (1 + energy):
            if self.alpha_conf > 1:
                self.alpha_conf /= 2
            if self.alpha_rigid > 1:
                self.alpha_rigid /= 2
            if self.alpha_smooth > 0.1:
                self.alpha_smooth /= 2

        self.last_energy = energy
        
        return energy

    
    def Esmooth(self, A_matrices, b_vectors, confidences):
        """Computes the energy that is minimized when adjacent
        vertices have similar transformations. 

        trimesh    - the mesh over which to compute the energy.

        A_matrices - an array of transformation matrices Ai, 
        corresponding to the vertex referenced by trimesh.vs[i].

        b_vectors  - an array of translation vectors bi,
        corresponding to the vertex referenced by trimesh.vs[i]."""

        total = 0.0
        for i in range(len(self.source_mesh.vs)):
            xi  = self.source_mesh.vs[i]

            js  = self.source_mesh.vertex_vertex_neighbors(i)

            Ai  = A_matrices[i]
            bi  = b_vectors[i]

            # for each neighbor
            for j in js:
                xj = self.source_mesh.vs[j]
                bj = b_vectors[j]

                # see page 4 of Hao Li, Esmooth = ...
                energy_vector = np.dot(Ai, xj - xi) + xi + bi - (xj + bj)

                # l2 norm squared is the same as dot(x^T, x)
                total += np.dot(energy_vector, energy_vector)

        return total

    
    def Efit(self, transformed_points, confidences):
        """Returns the energy associated with the fitting of source vertices
        to destination vertices.
        
        Unlike Li's paper, it uses a nearest-neighbor search to find error,
        rather than using the z-coordinate of the registration."""
        total = 0
        for i in range(len(self.source_mesh.vs)):
            total += confidences[i]**2 * nearest_neighbor_distance(transformed_points[i], self.destination_nearest_neighbors)**2

        return total

    def Efiteasy(self, transformed_points):
        total = 0
        for i in range(len(self.source_mesh.vs)):
            total += nearest_neighbor_distance(transformed_points[i], self.destination_nearest_neighbors)**2

        return total


    
    def Econf(self, confidences):
        """Returns the energy which penalizes low confidences."""
        return sum((1 - conf**2)**2 for conf in confidences)

    
    def Erigid(self, A_matrices):
        """Computes the energy that is minimized when the
        total transformation is rigid in nature; penalizes
        nonrigid transformations. A_matrices is an array
        of transformation matrices, one per vertex of the trimesh."""
        def Rot(A):
            """Returns a helper matrix Rot(A) from Li's paper.  A is a 3x3
            affine transformation matrix in proper form - columns as columns
            and rows as rows."""

            A01 = np.dot(A[:,0], A[:,1])
            A02 = np.dot(A[:,0], A[:,2])
            A12 = np.dot(A[:,1], A[:,2])

            A00 = np.dot(A[:,0], A[:,0])
            A11 = np.dot(A[:,1], A[:,1])
            A22 = np.dot(A[:,2], A[:,2])

            # Equation for Rot(A) on page 4 of Li's paper
            return A01 + A02 + A12 + (1.0 - A00)**2 + (1.0 - A11)**2 + (1.0 - A22)**2

        return sum(Rot(A) for A in A_matrices)

    
    def Egrasp(self, R, t, A_matrices, b_vectors, confidences):
        """Returns the energy which penalizes non-identification of the grasp
        points, if any are specified."""
        if not self.source_fixed_index:
            return 0.0

        i = self.source_fixed_index

        source_fixed_tr = LiAlgorithm.transform_vertex(self.source_fixed, 
                R, t, self.source_center_of_mass, A_matrices[i], b_vectors[i])

        return dist(source_fixed_tr, self.destination_fixed)

    def flatten_affine_transform(A):
        """Returns a flat np.array containing the important 12 components of
        the affine transform A."""
        return np.reshape(A[0:3, 0:4], 12)

    def unflatten_affine_transform(arr):
        result = np.identity(4)
        result[0:3,0:4] = np.reshape(arr, (3,4))
        return result

    def unpack_flat_array(self, arr):
        global_rot, global_trans = rotation_matrix(*arr[0:3]), arr[3:6]        

        A_matrices  = []
        b_vectors   = []
        confidences = []
        for i in range(len(self.source_mesh.vs)):
            curr = self.affine_of_vertex(arr, i).reshape((3,4))
            curr_transform = np.dot(global_rot, curr[0:3,0:3].reshape((3,3)))

            A_matrices.append(curr_transform)
            b_vectors.append(curr[0:3,3].reshape(3))
            confidences.append(self.conf_of_vertex(arr, i))

        return global_rot, global_trans, A_matrices, b_vectors, confidences

    def get_guess(self):
        arr = np.zeros(6 + len(self.source_mesh.vs)*13)

        if self.source_fixed is not None:
            arr[3:6] = self.destination_fixed - self.source_fixed

        for i in range(len(self.source_mesh.vs)):
            curr = LiAlgorithm.affine_of_vertex(arr, i).reshape((3,4))
            for j in range(3):
                curr[j,j] = 1

        return arr

    def run(self):
        guess = self.get_guess()
        result = minimize(self.energy, guess, options = {'disp': True, 'eps': 1e0, 'maxiter':20})
        (self.global_rot, self.global_trans, self.A_matrices, self.b_vectors,
                self.confidences) = self.unpack_flat_array(result.x)

    def affine_of_vertex(arr, i):
        """Returns the subarray corresponding to the affine transform of the ith
        vertex in arr."""
        # 6 global components,
        # 13 per vertex, stored
        # as [...a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 aconf...]
        start = 6 + i * 13
        return arr[start:start+12]

    affine_of_vertex = staticmethod(affine_of_vertex)

    def conf_of_vertex(arr, i):
        """Returns the confidence corresponding to the ith vertex in arr."""
        return arr[6 + i * 13 + 12]

    conf_of_vertex = staticmethod(conf_of_vertex)

    def transform_vertex(vertex, R, t, center_of_mass, A, b):
        """Transforms the vertex by the global rotation and translation, R and t,
        and then a local rotation and translation, A and b."""
        local = np.dot(A, vertex) + b
        return np.dot(R, local - center_of_mass)+ center_of_mass + t

    transform_vertex = staticmethod(transform_vertex)

    def transform(source_point):
        index = nearest_neighbor_index(source_point, self.source_nearest_neighbors)
        result = transform_vertex(self.global_rot, self.global_trans,
                self.A_matrices[index], self.b_vectors[index])
        
        return result



# for interpolating between multiple points, if we ever get around to that...
def weights_of_nearest(v, knearest, dmax):
    """Returns the weights associated with each of the k nearest nodes to
    queried vertex v, knearest, for the purposes of weighting in
    transformations."""

    # dmax is the distance from the k+1th nearest node to v;
    # i.e. knearest are all closer to v than dmax.

    # from paper:                       
    #                                   
    #                                   
    # wi(vj) =     1 - ||vj - xi||/dmax 
    #              -------------------- 
    #         sum (1 - ||vj - xp||/dmax)
    #         p=1..k                    
    numerators  = [1 - num.linalg.norm(v - xi, 2)/dmax for xi in knearest]
    denominator = sum(numerators)

    return [num / denominator for num in numerators]

def transform_v(A, b, v):
    return np.dot(A, v) + b
