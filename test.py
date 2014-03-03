from objfile import *
from sys import argv
import numpy as np
from nns import *
import unittest

class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.point = np.array([1,0,0,1])
        self.points = np.array([[1,0,0,1],
                                [0,1,0,1],
                                [0,0,1,1]])

    def test_point_translation(self):
        translated_point = np.dot(translation_matrix(np.array([1,1,1])), self.point)

        self.assertTrue(np.array_equal(translated_point, np.array([2,1,1,1])))

    def test_cloud_translation(self):
        translated_points = apply_transform(translation_matrix(np.array([1,1,1])), self.points)

        self.assertTrue(np.array_equal(translated_points, np.array([[2,1,1,1],
                                                                    [1,2,1,1],
                                                                    [1,1,2,1]])))


def test_icp(P, iterations):
    for i in range(iterations):
        # M = promote(np.random.random((3,3)))
        M = promote(rotation_matrix(0,math.pi / 2, 0))

        print "Actual transform:"
        print M

        P_copy = P.copy()
        print "P:"
        print P
        P_copy = apply_transform(M, P_copy)
        print "X:"
        print P_copy
        icp_estimate_M = icp(P, P_copy)

        print "Predicted transform:"
        print icp_estimate_M

        print "Error:", mean_square_error(M, icp_estimate_M)
        print "Error in identity:", mean_square_error(M, np.identity(4))

if __name__ == "__main__":
    # x = load_obj_file(argv[1])
    # points = x[1]

    #points = np.array([[0,1,2,1],[2,3,4,1],[5,6,7,1]])
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGeometry)
    unittest.TextTestRunner(verbosity=2).run(suite)

    #test_icp(points, 1)
