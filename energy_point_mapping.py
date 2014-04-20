"""
energy_point_mapping.py

Defines the SimpleEnergyAlgorithm class which inherits from RegistrationAlgorithm, which
implements a simple energy minimization-based approach for registration between two point clouds.
"""
from point_mapping import *
import numpy as np
from scipy.optimize import minimize, leastsq
from geometry import *

import random

from operator import add
from objfile import *
import trimesh

from icp_point_mapping import scaling_step

VERBOSE = True
MAX_SAMPLE_SIZE = 1000

class SimpleEnergyAlgorithm(FixedPairRegistrationAlgorithm):
    def __init__(self, source_mesh, destination_mesh, 
                 source_fixed_point = None, destination_fixed_point = None, 
                 max_iterations = 10):
        super(SimpleEnergyAlgorithm, self).__init__(source_mesh,
                destination_mesh, source_fixed_point, destination_fixed_point)
        self.source_nearest_neighbors = NearestNeighbors(n_neighbors=1, 
                algorithm="kd_tree").fit(source_mesh.vs)
        self.destination_nearest_neighbors = NearestNeighbors(n_neighbors=1, 
                algorithm="kd_tree").fit(destination_mesh.vs)

        self.iteration = 1
        self.max_iterations = max_iterations

    def Efit(self, transformed_points):
        sample = random.sample(transformed_points, 
                               min(len(transformed_points), MAX_SAMPLE_SIZE)
                 )
        return sum(nearest_neighbor_distance(pt, self.destination_nearest_neighbors)**2 for pt in sample)

    def energy(self, arr):
        global_rot, global_tr = self.unpack_flat_array(arr)
        transformed_points = np.dot(global_rot,
                trimesh.asarray(self.source_mesh.vs).T).T + global_tr

        fit_energy = self.Efit(transformed_points)

        if VERBOSE:
            print 'Iteration ({0} / {1}) Current SimpleEnergy: {2}'.format(self.iteration, 
                                                                     self.max_iterations, 
                                                                     fit_energy)

        self.iteration += 1

        return fit_energy

    def unpack_flat_array(self, arr):
        global_rot, global_trans = rotation_matrix(*arr[0:3]), arr[3:6]        
        return global_rot, global_trans

    def get_guess(self):
        arr = np.zeros(6)
        return arr

    def run(self):
        scale = np.asmatrix(scaling_step(self.destination_mesh.vs, self.source_mesh.vs)[0:3,0:3])

        source_as_mat = np.asmatrix(trimesh.asarray(self.source_mesh.vs))
        source_as_mat *= scale

        self.source_mesh.vs = source_as_mat
        
        guess = self.get_guess()
        result = minimize(self.energy, guess, method = 'Nelder-Mead', options = {'disp': True,
            'maxiter': self.max_iterations})
        self.global_rot, self.global_tr = self.unpack_flat_array(result.x)

    def transform(self, source_point):
        return (np.dot(self.global_rot, source_point) + self.global_tr, 1.0)

class EnergyAlgorithm(FixedPairRegistrationAlgorithm):
    def __init__(self, source_mesh, destination_mesh, 
                 source_fixed_point = None, destination_fixed_point = None, 
                 max_iterations = 10):

        super(EnergyAlgorithm, self).__init__(source_mesh,
                destination_mesh, source_fixed_point, destination_fixed_point)
        self.source_nearest_neighbors = NearestNeighbors(n_neighbors=1, 
                algorithm="kd_tree").fit(source_mesh.vs)
        self.destination_nearest_neighbors = NearestNeighbors(n_neighbors=1, 
                algorithm="kd_tree").fit(destination_mesh.vs)

        self.iteration = 1
        self.max_iterations = max_iterations

        self.rows = len(self.source_mesh.vs)

    def Efit(self, transformed_points):
        sample = random.sample(transformed_points, 
                               min(len(transformed_points), MAX_SAMPLE_SIZE)
                 )
        return sum(nearest_neighbor_distance(pt, self.destination_nearest_neighbors)**2 for pt in sample)

    def energy(self, arr):
        global_rot, global_tr, local_tr = self.unpack_flat_array(arr)
        transformed_points = np.dot(global_rot,
                trimesh.asarray(self.source_mesh.vs).T).T + global_tr + local_tr


        fit_energy = self.Efit(transformed_points)

        if VERBOSE:
            print 'Iteration ({0} / {1}) Current Energy: {2}'.format(self.iteration, 
                                                                     self.max_iterations, 
                                                                     fit_energy)
        self.iteration += 1

        return fit_energy
        return np.array(transformed_points)

    def unpack_flat_array(self, arr):
        global_rot, global_tr = rotation_matrix(*arr[0:3]), arr[3:6]        
        local_tr = arr[6:].reshape(self.rows, 3)
        return global_rot, global_tr, local_tr

    def get_guess(self):
        arr = np.zeros(6 + len(self.source_mesh.vs) * 3)
        return arr

    def run(self):
        scale = np.asmatrix(scaling_step(self.destination_mesh.vs, self.source_mesh.vs)[0:3,0:3])

        source_as_mat = np.asmatrix(trimesh.asarray(self.source_mesh.vs))
        source_as_mat *= scale

        self.source_mesh.vs = source_as_mat
        
        guess = self.get_guess()
        result = minimize(self.energy, guess)
        self.global_rot, self.global_tr = self.unpack_flat_array(result.x)

    def transform(self, source_point):
        return (np.dot(self.global_rot, source_point) + self.global_tr, 1.0)
