import numpy as np
from sys import argv

def load_obj_file(path):
    """Loads the vertices of a .obj file into a np array as 3-vectors.
    Ignores the information about connectivity and other metadata."""
    points = []
    line_no = 0
    try:
        with open(path, "r") as f:
            for line in f.readlines():
                line_no += 1
                line = line.strip()

                splits = line.split(' ')
                if splits[0] != "v": continue # if this is NOT a vertex row
                try:
                    (x, y, z) = (float(splits[1]), 
                                 float(splits[2]), 
                                 float(splits[3]))

                    points.append([x,y,z])
                except Exception as e:
                    if len(splits) != 4:
                        print ("Error while loading line {0:>5d}: "
                            "Missing an expected parameter (x, y, or z). '{1}'"
                            ).format(line_no, line)
                    else:
                        print ("Error while loading line {0:>5d}: '{1}'"
                            ).format(line_no, line)
    except Exception as e:
        print e
        return None

    return np.array(points)

if __name__ == "__main__":
    if len(argv) > 1:
        for x in argv[1:]:
            print repr(load_obj_file(x))
