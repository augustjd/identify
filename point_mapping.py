#!/usr/bin/env python

from sklearn.neighbors import NearestNeighbors
import numpy as np 

from trimesh import TriMesh
from geometry import dist, estimate_max_diagonal, transform_trimesh

def clamp(val, low, high):
    """Clamps val to the range [low, high]."""
    return max(min(val, high), low)

class RegistrationAlgorithm(object):
    def __init__(self, source_mesh, destination_mesh):
        assert(isinstance(source_mesh, TriMesh) and 
               isinstance(destination_mesh, TriMesh))

        self.source_mesh      = source_mesh 
        self.destination_mesh = destination_mesh 

        self.global_confidence = 1.0

        self.destination_nearest_neighbors = (
                NearestNeighbors(n_neighbors=1,
                    algorithm="kd_tree").fit(destination_mesh.vs))

        self.destination_longest_diagonal = (
                estimate_max_diagonal(destination_mesh.vs))

    def run(self):
        pass

    def transform(self, source_point):
        """In classes which inherit from RegistrationAlgorithm, this
        function will perform the mapping on source_point, and return
        the tuple (predicted_point, confidence)."""
        pass

    def register(self, mapping):
        """Finds a registration for mapping.source on the destination_mesh,
        and also stores in mapping.confidence the confidence in that mapping."""
        assert(isinstance(mapping, PointMapping))

        mapping.destination, mapping.confidence = self.transform(mapping.source)
        mapping.destination, distance           = self.project(mapping.destination)

        mapping.confidence *= self.get_confidence_of_projection(distance)
        mapping.confidence *= self.global_confidence

    def project(self, source_point):
        """Finds the closest point on the destination mesh for the given
        source_point, and returns the projected point as well as the distance
        to that point."""
        projected_point_index = self.destination_nearest_neighbors.kneighbors(
                                    source_point
                                )[1][0][0]
        projected_point = self.destination_mesh.vs[projected_point_index]

        return projected_point, dist(projected_point, source_point)

    def get_confidence_of_projection(self, distance, tolerance = 0.5):
        """Returns a confidence metric 0.0 < c < 1.0 corresponding to a
        projection in which a point P was moved to a point Q that is dist
        away."""
        raw = 1.0 - (distance / self.destination_longest_diagonal)**(tolerance)
        return clamp(raw, 0.0, 1.0)

    def transformed_mesh(self):
        """Transforms a copy of the source_mesh by the transformation."""
        result = self.source_mesh.copy()

        # but transform returns a tuple, so...
        def one_return_transform(x):
            return self.transform(x)[0]

        transform_trimesh(result, one_return_transform)

        return result

class PointMapping:
    """Encapsulates the mapping of a point of interest to another point of
    interest, for use in recording the output of registration algorithms."""

    def __init__(self, label, source, destination = None, confidence = None):
        self.label       = label
        self.source      = source 
        self.destination = destination 
        self.confidence  = confidence

    def __str__(self):
        """Returns the string representation of this PointMapping in the
        format 'lbl x-src y-src z-src x-dst y-dst z-dst conf'."""
        return "{0} {1:8f} {2:8f} {3:8f} {4:8f} {5:8f} {6:8f} {7:8f}".format(
                    self.label,
                    self.source[0], 
                    self.source[1], 
                    self.source[2],
                    self.destination[0], 
                    self.destination[1], 
                    self.destination[2],
                    self.confidence
                )

    def to_file(path, mappings):
        """Saves an array of PointMappings to a point mapping file specified
        in path."""
        try:
            with open(path, "w") as f:
                for i, point in enumerate(mappings):
                    if i != 0:
                        f.write("\n")

                    f.write(str(point))

        except Exception as e:
            print e
            return None

    to_file = staticmethod(to_file)

    def from_file(path):
        """Parses an array of PointMappings from a point mapping file, and
        returns that array. If errors occur in parsing, whatever points were
        successfully parsed will be returned."""
        points = []

        try:
            with open(path, "r") as f:
                for i, line in enumerate(f.readlines()):
                    splits = line.split()
                    try:
                        label  = splits[0]
                        source = np.array([float(splits[1]), 
                                           float(splits[2]), 
                                           float(splits[3])])

                        points.append(PointMapping(label, source))

                    except Exception as err:
                        print "Failed to parse line {0}: '{1}'".format(i, line)

        except Exception as err:
            print err

        return points

    from_file = staticmethod(from_file)
