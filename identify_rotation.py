#!/usr/bin/env python
"""
identify_rotation.py
A simple script that, given two identical meshes,
but where one is rotated about the Y axis, estimates
the rotation in radians, and outputs it.
"""

from radial_point_mapping import RadialAlgorithm
from radial_point_mapping import TriMesh
from radial_point_mapping import estimate_grasp_point
import numpy as np

from sys import argv
# import argparse

def print_usage():
    """Prints the usage for identify_rotation.py."""
    print "Usage: {0} <source_file> <destination_file>".format(
            argv[0]
            )
    print "{0} determines the rotation needed to take source to destination."
    print "Given two identical meshes, where source is rotated about the "
    print "Y axis at the origin theta degrees, outputs theta."

TEST = True
MINIMAL_OUTPUT = True

def estimate_theta_of_rotation(source_mesh, destination_mesh):
    """Estimates the rotation about the Y axis at the origin needed to take
    source_mesh to destination_mesh, returning theta in radians."""

    if not MINIMAL_OUTPUT:
        print "Estimating grasp points..."
    grasp_points = (estimate_grasp_point(source_mesh.vs), 
                    estimate_grasp_point(destination_mesh.vs))

    if not MINIMAL_OUTPUT:
        print "Grasp points:\n\tSource:\t\t{0}\n\tDestination:\t{1}".format(
                grasp_points[0],
                grasp_points[1])

    source_mesh.vs -= grasp_points[0]
    destination_mesh.vs -= grasp_points[1]

    grasp_points = (estimate_grasp_point(source_mesh.vs), 
                    estimate_grasp_point(destination_mesh.vs))

    algo = RadialAlgorithm(
            source_mesh, destination_mesh,
            grasp_points[0], grasp_points[1]
        )

    algo.run()
    if not MINIMAL_OUTPUT:
        print "Best theta estimate: {0} Point Error: {1}".format(
                algo.theta_estimate,
                algo.error)

    return algo.theta_estimate

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(prog = 'identify_rotation')

    if len(argv) != 3:
        print_usage()
        exit(0)

    try:
        SOURCE = TriMesh.FromOBJ_FileName(argv[1])
    except AssertionError:
        print "Failed to load source: Something is wrong with the source mesh."
        exit(0)

    try:
        DESTINATION = TriMesh.FromOBJ_FileName(argv[2])
    except AssertionError:
        print "Failed to load source: Something is wrong with the source mesh."
        exit(0)

    # if TEST:
    #     print "TESTING"
    #     import random, math
    #     import geometry
    #     theta = random.random() * math.pi * 2
    #     print "Rotating by {0} radians...".format(theta)
    #     rot_matrix = geometry.single_rotation_matrix(theta, 1)
    #     SOURCE.vs = np.dot(SOURCE.vs, rot_matrix)

    #     print estimate_theta_of_rotation(SOURCE, DESTINATION)

    #     print "Actual theta: {0}".format(theta)
    #     exit(0)

    print estimate_theta_of_rotation(SOURCE, DESTINATION)
