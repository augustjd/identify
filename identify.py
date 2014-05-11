#!/usr/bin/env python

import numpy as np
from icp_point_mapping import *
from li_point_mapping import *
from energy_point_mapping import *
from radial_point_mapping import *
from curvature_point_mapping import *

def print_usage():
    print "Usage: {0} [flags] <source_file> <destination_file> <point_set_file> <output_file>".format(
            argv[0]
            )
    print
    print "Supported Flags"
    print "\t-h:\t\t\tprints this help message, then exits."
    print
    print "\t-no-point-file:\t\tDon't read or write to point set files, so the final two"
    print "\t\t\t\targuments can be omitted from the invocation. Good if you"
    print "\t\t\t\tjust want a mesh file output. This makes invocation look like:"
    print
    print "\t\t\t\t{0} -no-point-file -m [mesh_output_file] [flags] <source_file> <destination_file>".format(
            argv[0]
            )
    print
    print "\t-m [mesh_output_file]:\tmesh output mode - in addition to a point set file, save the"
    print "\t\t\t\tentire cloud under the mapping as a .obj file to [mesh_output_file]."
    print
    print "\t-projectm [mesh_output_file]:\tprojected mesh output mode - in addition to a point set file, save the"
    print "\t\t\t\tentire cloud under the mapping, projected onto the destination as a .obj file to "
    print "\t\t\t\t[mesh_output_file]."
    print
    print ("\t-i [s-index] [d-index]:\tensure that the mapping identifies the " +
        "point of index s-index on")
    print "\t\t\t\tS and of d-index on D. By default, the centers of mass of S"
    print "\t\t\t\tand D are identified."
    print
    print "\t-v:\t\t\tverbose output mode"
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
    match = last_flag_matching(flags, key)
    if match is not None:
        match_index = flags.index(match)
        return flags[match_index + 1 : match_index + argc + 1]
    else:
        return None

def all_flag_args(flags, key, argc):
    """Searches flags for the flags matching key. For each match,
    the subsequent argc args will be provided."""
    match_indices = filter(lambda i: flags[i] == key, range(len(flags)))
    if argc == 1:
        return [flags[i + 1] for i in match_indices]
    elif argc > 1:
        return [flags[i + 1 : i + argc + 1] for i in match_indices]

VALID_ALGORITHMS = {
        'icp':       IcpAlgorithm,
        'li':        LiAlgorithm,
        'simple':    SimpleEnergyAlgorithm,
        'energy':    EnergyAlgorithm,
        'radial':    RadialAlgorithm,
        'curvature': CurvatureAlgorithm,
        'curvatureicp': CurvatureIcp
        }

from sys import *
def main():
    flags = argv[:-3]
    
    if "-no-point-file" in flags:
        flags = argv[:-2]

    if len(argv) < 4 or "-h" in flags:
        print_usage()
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

    mesh_output_file_names = all_flag_args(flags, "-m", 1)    
    if "-m" in flags:
        print "Mesh output mode. Outputting registered meshes to {0}.".format(
                ",".join(mesh_output_file_names))

    projected_mesh_output_file_names = all_flag_args(flags, "-projectm", 1)    
    if "-projectm" in flags:
        print "Projected mesh output mode. Outputting registered meshes to {0}.".format(
                ",".join(mesh_output_file_names))

    if "-no-point-file" in flags:
        source_mesh_file_name       = argv[-2]
        destination_mesh_file_name  = argv[-1]
        point_set_file_name         = None
        output_file_name            = None
    else:
        source_mesh_file_name       = argv[-4]
        destination_mesh_file_name  = argv[-3]
        point_set_file_name         = argv[-2]
        output_file_name            = argv[-1]
    
    try:
        source_mesh      = TriMesh.FromOBJ_FileName(source_mesh_file_name)
        print "Source mesh loaded with {0} vertices.".format(len(source_mesh.vs))
    except AssertionError:
        print "Failed to load source: Something is wrong with the source mesh."
        exit(0)

    try:
        destination_mesh = TriMesh.FromOBJ_FileName(destination_mesh_file_name)
        print "Destination mesh loaded with {0} vertices.".format(len(destination_mesh.vs))
    except AssertionError:
        print "Failed to load destination: Something is wrong with the destination mesh."
        exit(0)

    grasp_points = (None,None)
    if point_set_file_name is not None:
        mappings, grasp_points = PointMapping.from_file(point_set_file_name, source_mesh)

    if grasp_points == (None,None):
        print "Estimating grasp points..."
        grasp_points = (estimate_grasp_point(source_mesh.vs), 
                        estimate_grasp_point(destination_mesh.vs))

    print "Grasp points:\n\tSource:\t\t{0}\n\tDestination:\t{1}".format(*grasp_points)
    source_grasp_point = grasp_points[0]
    transformed = source_mesh.copy()

    verbose = "-v" in flags
    if verbose:
        print "Verbose mode."

    for i, algo in enumerate(algorithms_list):
        iteration = algo(
            source_mesh, destination_mesh,
            source_grasp_point, grasp_points[1],
            verbose=verbose
        )

        iteration.run()

        transformed = iteration.transformed_mesh()
        print "Finished transformation {0}/{1}.".format(i+1,
                len(algorithms_list))

    for path in mesh_output_file_names:
        # use the red texture that's already in testdata/
        transformed.write_OBJ(path, "mtllib default.mtl\nusemtl defaultred")

        print "Wrote mesh to '{0}'.".format(path)

    if len(projected_mesh_output_file_names) > 0:
        projected = iteration.projected_mesh()
        for path in projected_mesh_output_file_names:
            # use the red texture that's already in testdata/
            projected.write_OBJ(path, "mtllib default.mtl\nusemtl defaultred")

            print "Wrote mesh to '{0}'.".format(path)

    if (len(mesh_output_file_names) > 0 or 
            len(projected_mesh_output_file_names) > 0):
        print "Compare with '{0}'.".format(destination_mesh_file_name)

    if output_file_name is not None:
        PointMapping.to_file(output_file_name, mappings)

if __name__ == "__main__":
    main()
    #import cProfile
    #print cProfile.run('main()')

