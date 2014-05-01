#!/usr/bin/env python

import numpy as np
from icp_point_mapping import *
from li_point_mapping import *
from energy_point_mapping import *
from radial_point_mapping import *
from curvature_point_mapping import *

def print_usage():
    print "Point Set Mode Usage:"
    print "{0} [flags] <source_file> <destination_file> <point_set_file> <output_file>".format(
            argv[0]
            )
    print
    print "Mesh output mode usage:"
    print "{0} -m [flags] <source_file> <destination_file> <output_mesh_file>".format(
            argv[0]
            )
    print
    print "Supported Flags"
    print "\t-h:\t\t\tprints this help message, then exits."
    print
    print "\t-m:\t\t\tmesh output mode - instead of a point set file, save the"
    print "\t\t\t\tentire cloud under the mapping as a .obj file to <output_file>."
    print "\t\t\t\t<point_set_file> must be present, but will be ignored."
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
    print "\t\t\t\tTo use several in sequence, write them in order delimited by +,"
    print "\t\t\t\tsuch as icp+li."
    print "\t\t\t\tvalid options: {0}".format(", ".join(VALID_ALGORITHMS.keys()))

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
        'icp':       IcpAlgorithm,
        'li':        LiAlgorithm,
        'simple':    SimpleEnergyAlgorithm,
        'energy':    EnergyAlgorithm,
        'radial':    RadialAlgorithm,
        'curvature': CurvatureAlgorithm,
        'curvatureicp': CurvatureIcp
        }

if __name__ == "__main__":
    from sys import *
    flags = argv[:-3]

    if len(argv) < 4 or "-h" in flags:
        print_usage()
        exit(0)

    if "-v" in flags:
        print "Verbose."
        VERBOSE_DEFAULT = True

    new_threshold = flag_value(flags, "--convergence")
    if new_threshold:
        try:
            CONVERGENCE_THRESHOLD = float(eps)
        except Exception as e: 
            print "Error parsing convergence threshold {0}".format(eps) 
            exit(0)

    algorithm_name = "icp"

    new_algorithm = flag_value(flags, "--algorithm")
    algorithms_list = []
    if new_algorithm:
        algorithm_names = new_algorithm.split("+")
        for name in algorithm_names:
            if name in VALID_ALGORITHMS:
                algorithms_list.append(VALID_ALGORITHMS[name])
            else:
                print "Algorithm '{0}' is not supported.".format(new_algorithm)
                print "Valid options: {0}".format(", ".join(VALID_ALGORITHMS.keys()))
                exit(0)
        print "Performing in order:", "->".join(algorithm_names)
    else:
        algorithms_list = [IcpAlgorithm]

    ux_index = None
    up_index = None
    
    index_args = flag_args(flags, "-i", 2)
    if index_args:
        ux_index = int(index_args[0]) # source
        up_index = int(index_args[1]) # destination

    if "-m" in flags:
        source_mesh_file_name = argv[-3]
        destination_mesh_file_name = argv[-2]
    else:
        source_mesh_file_name = argv[-4]
        destination_mesh_file_name = argv[-3]

    output_file_name = argv[-1]
    
    try:
        source_mesh      = TriMesh.FromOBJ_FileName(source_mesh_file_name)
    except AssertionError:
        print "Failed to load source: Something is wrong with the source mesh."
        exit(0)

    try:
        destination_mesh = TriMesh.FromOBJ_FileName(destination_mesh_file_name)
    except AssertionError:
        print "Failed to load source: Something is wrong with the source mesh."
        exit(0)

    grasp_points = (None,None)
    if "-m" not in flags:
        mappings, grasp_points = PointMapping.from_file(argv[-2], source_mesh)

    if grasp_points == (None,None):
        print "Estimating grasp points..."
        grasp_points = (estimate_grasp_point(source_mesh.vs), 
                        estimate_grasp_point(destination_mesh.vs))

    print "Grasp points:\n\tSource:\t\t{0}\n\tDestination:\t{1}".format(*grasp_points)
    source_grasp_point = grasp_points[0]
    transformed = source_mesh.copy()

    if "-m" in flags:
        print "Mesh output mode."
        for i, algo in enumerate(algorithms_list):
            iteration = algo(
                source_mesh, destination_mesh,
                source_grasp_point, grasp_points[1]
            )

            iteration.run()

            transformed = iteration.transformed_mesh()
            print "Finished transformation {0}/{1}.".format(i+1,
                    len(algorithms_list))

        # use the red texture that's already in testdata/
        transformed.write_OBJ(output_file_name, "mtllib default.mtl\nusemtl defaultred")

        print "Wrote mesh to '{0}'.".format(output_file_name)
        print "Compare with '{0}'.".format(destination_mesh_file_name)

    else: # default mode
        for i, algo in enumerate(algorithms_list):
            iteration = algo(
                source_mesh, destination_mesh,
                source_grasp_point, grasp_points[1]
            )

            iteration.run()

            transformed = iteration.transformed_mesh()

            for mapping in mappings:
                iteration.register(mapping)

            # move source fixed point
            source_grasp_point, _ = iteration.transform(source_grasp_point)

            print "Finished transformation {0}/{1}.".format(i+1,
                    len(algorithms_list))

        PointMapping.to_file(output_file_name, mappings)
