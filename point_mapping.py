#!/usr/bin/env python

import numpy as np 

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

    def self.transformList(transformation, mappings):
        """Applies the function transformation to each point in mappings,
        saving the result in its (destination, confidence)."""
        for x in mappings:
            x.destination, x.confidence = transformation(x.source)
