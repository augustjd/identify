#!/usr/bin/env python

from sklearn.neighbors import NearestNeighbors
import numpy as np 

from trimesh import *
from geometry import *

def clamp(val, low, high):
    return max(min(val, high), low)

class RegistrationAlgorithm:
    def __init__(self, source_mesh, destination_mesh):
        assert(isinstance(self.source_mesh, TriMesh) and 
               isinstance(self.destination_mesh, TriMesh))

        self.source_mesh      = source_mesh 
        self.destination_mesh = destination_mesh 

        self.global_confidence = 1.0

        self.destination_nearest_neighbors = 
            NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(destination_mesh.vs)

        self.destination_longest_diagonal =
            estimate_max_diagonal(destination_longest_diagonal.vs)

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

        mapping.destination, mapping.confidence = transform(mapping.source)
        mapping.destination, distance           = project(mapping.destination)

        mapping.confidence *= get_confidence_from_projection_distance(distance)

        mapping.confidence *= self.global_confidence

    def project(self, source_point):
        """Finds the closest point on the destination mesh for the given
        source_point, and returns the projected point as well as the distance
        to that point."""
        projected_point = self.destination_nearest_neighbors.kneighbors(np.array(source_point))

        return projected_point, dist(projected_point, source_point)

    def get_confidence_from_projection_distance(self, dist, K = 0.5):
        """Returns a confidence metric 0.0 < c < 1.0 corresponding to a
        projection in which a point P was moved to a point Q that is dist
        away.  Parameter K > 0 affects the tolerance - as it gets smaller,
        the confidence associated with medium distances becomes smaller."""
        raw = 1.0 - (dist / self.destination_longest_diagonal)**(K)
        return clamp(raw, 0.0, 1.0)

class PointMapping:
    def __init__(self, label, source, destination = None, conf = None):
        self.label       = label
        self.source      = source 
        self.destination = destination 
        self.confidence  = confidence

    def __str__(self):
        """Returns the string representation of this PointMapping in the
        format 'lbl x-src y-src z-src x-dst y-dst z-dst conf'."""
        return "{0} {1:8f} {2:8f} {3:8f} {4:8f} {5:8f} {6:8f} {7:8f}".format(
                    self.label,
                    self.source[0],   self.source[1],   self.source[2],
                    self.destination[0], self.destination[1], self.destination[2],
                    self.confidence
                )

    def self.toFile(path, mappings):
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

    def self.fromFile(path):
        """Parses an array of PointMappings from a point mapping file, and
        returns that array. If errors occur in parsing, whatever points were
        successfully parsed will be returned."""
        points = []

        try:
            with open(path, "r") as f:
                for i, line in enumerate(f.readlines()):
                    splits = line.split(" ")
                    try:
                        label  = splits[0]
                        source = np.array([float(splits[1]), 
                                           float(splits[2]), 
                                           float(splits[3])])

                        points.append(PointMapping(label, source))

                    except IndexError as e:
                        print "Failed to parse line {0}: '{1}'".format(i, line)

        except Exception as e:
            print e

        return points
