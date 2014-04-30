"""
energy_point_mapping.py

Defines the SimpleEnergyAlgorithm class which inherits from RegistrationAlgorithm, which
implements a simple energy minimization-based approach for registration between two point clouds.
"""
from point_mapping import *
import numpy as np
from scipy.optimize import minimize, leastsq, fmin_bfgs
from geometry import *

import random

from operator import add
from objfile import *
import trimesh

from icp_point_mapping import scaling_step

VERBOSE = True
MAX_SAMPLE_SIZE = 500

class SimpleEnergyAlgorithm(FixedPairRegistrationAlgorithm):
    def __init__(self, source_mesh, destination_mesh, 
                 source_fixed_point = None, destination_fixed_point = None, 
                 max_iterations = 300):
        super(SimpleEnergyAlgorithm, self).__init__(source_mesh,
                destination_mesh, source_fixed_point, destination_fixed_point)
        self.source_nearest_neighbors = NearestNeighbors(n_neighbors=1, 
                algorithm="kd_tree").fit(source_mesh.vs)
        self.destination_nearest_neighbors = NearestNeighbors(n_neighbors=1, 
                algorithm="kd_tree").fit(destination_mesh.vs)

        self.dnn = self.destination_nearest_neighbors

        self.iteration = 1
        self.max_iterations = max_iterations

    def Efit(self, transformed_points):
        sample = random.sample(transformed_points, 
                               min(len(transformed_points), MAX_SAMPLE_SIZE)
                 )
        return sum(nearest_neighbor_distance(pt, self.dnn)**2 for pt in sample) / len(sample)


    def energy(self, arr):
        global_rot, global_tr = self.unpack_flat_array(arr)
        transformed_points = self.source_mesh.vs + global_tr

        transformed_points = np.dot(global_rot,
                trimesh.asarray(self.source_mesh.vs).T).T + global_tr

        fit_energy = self.Efit(transformed_points)


        self.iteration += 1

        return fit_energy

    def unpack_flat_array(self, arr):
        global_rot, global_trans = rotation_matrix(*arr[0:3]), arr[3:6]        
        return global_rot, global_trans

    def get_guess(self):
        arr = np.zeros(6)
        return arr
    
    def print_energy(self, xk):
        global_rot, global_tr = self.unpack_flat_array(xk)
        print "Simple Energy: {0} Tr: {1} Rot: {2}".format(self.energy(xk),
                global_tr, global_rot)

    def run(self):
        scale = np.asmatrix(scaling_step(self.destination_mesh.vs, self.source_mesh.vs)[0:3,0:3])

        source_as_mat = np.asmatrix(trimesh.asarray(self.source_mesh.vs))
        source_as_mat *= scale

        self.source_mesh.vs = source_as_mat
        
        guess = self.get_guess()
        result = minimize(self.energy, guess, callback = self.print_energy,
                options = {'maxiter': self.max_iterations, 'eps': 1e-2, 'disp': True})
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

        self.scaling_factor = 30

    def Efit(self, transformed_points):
        sample = random.sample(transformed_points, 
                               min(len(transformed_points), MAX_SAMPLE_SIZE)
                 )
        return self.scaling_factor * sum(nearest_neighbor_distance(pt, self.destination_nearest_neighbors)**2 for pt in sample)

    def grad(f, x0, d = 1e-2):
        n       = len(x0)
        result  = np.zeros(n)
        dx      = np.zeros(n)

        for i in range(n):
            dx[i] = d
            result[i] = f(x0 + dx) - f(x0)
            dx[i] = 0

        result /= d
        return result 

    def energy(self, arr):
        global_rot, global_tr, local_tr = self.unpack_flat_array(arr)
        transformed_points = np.dot(global_rot,
                trimesh.asarray(self.source_mesh.vs).T).T + global_tr + local_tr


        fit_energy = self.Efit(transformed_points)

        print "Energy Function Invocation {0}: Energy {1}".format(self.iteration, fit_energy)

        self.iteration += 1

        return fit_energy

    def print_energy(self, x):
        print "Hello!"
        fit_energy = self.Efit(x)

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
        result = minimize(self.energy, guess, callback = self.print_energy,
                options = {'disp': True, 'eps': 1e0})
        self.global_rot, self.global_tr = self.unpack_flat_array(result.x)

    def transform(self, source_point):
        return (np.dot(self.global_rot, source_point) + self.global_tr, 1.0)
