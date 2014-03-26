#!/usr/bin/env python

from sklearn.neighbors import NearestNeighbors
import numpy as np
from icp_point_mapping import *

def print_usage():
    print "Usage: {0} [flags] <source_file> <destination_file> <point_set_file> <output_file>".format(
            argv[0]
            )
    print "Supported Flags"
    print "\t-h:\t\t\tprints this help message, then exits."
    print
    print "\t-m:\t\t\tmesh output mode - instead of a point set file, save the"
    print "\t\t\t\tentire cloud under the mapping as a .obj file to <output_file>"
    print
    print ("\t-i [s-index] [d-index]:\tensure that the mapping identifies the " +
        "point of index s-index on")
    print "\t\t\t\tS and of d-index on D. By default,the centers of mass of S "
    print "\t\t\t\tand D are identified."
    print
    print "\t-v:\t\t\tverbose output mode"
    print
    print "\t--convergence=[val]:\trun identification until matching confidence exceeds val (default: 0.1)"
    print
    print ("\t--algorithm=[val]:\tuse specified algorithm to perform " +
            "registration (default: icp).")
    print "\t\t\t\tvalid options: icp"

def flag_exists(flags, flag):
    return last_flag_matching(flags, flag) is not None

def last_flag_matching(flags, key):
    return next((flag for flag in reversed(flags) if flag.startswith(key)), None)

def flag_value(flags, key):
    """Returns the string value of the LAST flag in flags of form [key]=[val], or,
    if no such key exists, None."""
    key = key + "="
    match = last_flag_matching(flags, key)
    if match is not None:
        return match[len(key):]
    else:
        return None

def flag_args(flags, key, argc):
    """Searches flags for the last flag matching key. If one exists, returns
    it and the next argc arguments."""
    indices = reversed(range(len(flags)))
    match = last_flag_matching(flags, key)
    if match is not None:
        match_index = flags.index(match)
        return flags[match_index + 1 : match_index + argc + 1]
    else:
        return None

VALID_ALGORITHMS = {
        'icp': IcpAlgorithm
        }
if __name__ == "__main__":
    from sys import *
    flags = argv[:-3]

    if len(argv) < 4 or "-h" in flags:
        print_usage()
        exit(0)

    if "-v" in flags:
        print "Verbose."
        VERBOSE = True

    new_threshold = flag_value(flags, "--convergence")
    if new_threshold:
        try:
            CONVERGENCE_THRESHOLD = float(eps)
        except Exception as e: 
            print "Error parsing convergence threshold {0}".format(eps) 
            exit(0)

    algorithm_name = "icp"

    new_algorithm = flag_value(flags, "--algorithm")
    if new_algorithm:
        if new_algorithm in valid_algorithms:
            algorithm_name = new_algorithm

    ux_index = None
    up_index = None
    
    index_args = flag_args(flags, "-i", 2)
    if index_args:
        ux_index = int(index_args[0]) # source
        up_index = int(index_args[1]) # destination

    source_mesh      = TriMesh.FromOBJ_FileName(argv[-4])
    destination_mesh = TriMesh.FromOBJ_FileName(argv[-3])

    if "-m" in flags:
        icp_and_output_to_mesh(argv[-4], argv[-3], argv[-1])
    else: # default mode
        algo = VALID_ALGORITHMS[algorithm_name](
            source_mesh, destination_mesh,
            ux_index, up_index
        )

        mappings = PointMapping.from_file(argv[-2])

        algo.run()

        for mapping in mappings:
            algo.register(mapping)

        PointMapping.to_file(argv[-1], mappings)
        


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
