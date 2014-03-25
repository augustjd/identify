#!/usr/bin/env python

from sklearn.neighbors import NearestNeighbors
import numpy as np

if __name__ == "__main__":
    from sys import *
    if len(argv) < 4:
        print "Usage: {0} [flags] <source_file> <destination_file> <point_set_file> <output_file>".format(
                argv[0]
                )
        print "Supported Flags"
        print "\t-m:\t\t\tmesh output mode - instead of a point set file, save the"
        print "\t\t\t\tentire cloud under the mapping as a .obj file to <output_file>"
        print
        print ("\t-i [s-index] [d-index]:\tensure that the mapping identifies the" +
            "point of index s-index")
        print "\t\t\t\tS and of d-index on D. By default,the centers of mass of S "
        print "\t\t\t\tand D are identified."
        print
        print "\t-v:\t\t\tverbose output mode"
        print
        print "\t--convergence=[val]:\trun identification until matching confidence "
        print "\t\t\t\texceeds val (default 0.1)"
        exit(0)

    flags = argv[:-3]

    if "-v" in flags:
        print "Verbose."
        VERBOSE = True

    eps = filter(lambda s: s.startswith("--convergence="), flags)
    if len(eps) > 0:
        try:
            CONVERGENCE_THRESHOLD = float(eps[-1][len("--convergence="):])
        except Exception as e:
            print "Error parsing convergence threshold {0}".format(eps[-1])
            exit(0)

    if "-m" in flags:
        icp_and_output_to_mesh(argv[-4], argv[-3], argv[-1])
    else: # default mode
        icp_and_output_point_file(argv[-4], argv[-3], argv[-2], argv[-1])


def icp_and_output_to_mesh(source_path, destination_path, output_path):
    source_file_array = load_obj_file(source_path)
    print "Loaded {0} as {1} points.".format(source_path, len(source_file_array[1]))
    destination_mesh  = load_obj_file(destination_path)[1]
    print "Loaded {0} as {1} points.".format(destination_path, len(destination_mesh))

    source_mesh = source_file_array[1]

    P_nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(destination_mesh)
    transform = icp(destination_mesh, source_mesh, None, None, P_nearest_neighbors)[0]

    source_mesh = apply_transform(transform, source_mesh)
 
    if save_obj_file(output_path, 
                     source_file_array[0], 
                     source_mesh,
                     source_file_array[2]):
        print "Saved output to file '{0}'.".format(output_path)
    else:
        print "Failed to output to file '{0}'.".format(output_path)

def icp_and_output_point_file(source_path, destination_path,
                              point_set_file_path, output_path):

    source_file_array = load_obj_file(source_path)
    print "Loaded {0} as {1} points.".format(source_path, len(source_file_array[1]))
    destination_mesh  = load_obj_file(destination_path)[1]
    print "Loaded {0} as {1} points.".format(destination_path, len(destination_mesh))

    source_mesh = source_file_array[1]

    P_nearest_neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(destination_mesh)
    transform, error = icp(destination_mesh, source_mesh, None, None, P_nearest_neighbors)[0]
