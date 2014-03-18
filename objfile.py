import numpy as np 
from sys import argv

def load_obj_file(path):
    """Loads the vertices of a .obj file into a np array as 3-vectors.
    Returns an array consisting in all lines before vertex lines (header),
    a numpy array of all the vertices, and finally all lines after vertex 
    lines (footer)."""
    points = []
    header = ""
    footer = ""
    line_no = 0
    try:
        with open(path, "r") as f:

            for line in f.readlines():
                line_no += 1
                line = line.strip()

                splits = line.split(' ')
                if splits[0] != "v": # if this is NOT a vertex row
                    if len(points) == 0:
                        header += line + "\n"
                    else:
                        footer += line + "\n"
                    continue 

                try:
                    (x, y, z) = (float(splits[1]), 
                                 float(splits[2]), 
                                 float(splits[3]))

                    points.append([x,y,z,1])
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

    return [header, np.array(points), footer]

def save_obj_file(path, header, points, footer):
    try:
        with open(path, "w") as f:
            f.write(header)
            for p in points:
                f.write("v {0:7f} {1:7f} {2:7f}\n".format(p[0], p[1], p[2]))
            f.write(footer)

    except Exception as e:
        print e
        return None


if __name__ == "__main__":
    if len(argv) > 1:
        for x in argv[1:]:
            print repr(load_obj_file(x))
