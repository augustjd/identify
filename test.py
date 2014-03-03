from objfile import *
from sys import argv
import numpy as np
from nns import *
import unittest

class TestNumpyArray(unittest.TestCase):
    def assertArraysEqual(self, left, right):
        if type(left)  is list: left  = np.array(left)
        if type(right) is list: right = np.array(right)

        return self.assertTrue(np.array_equal(left, right))

    def assertArraysApproximatelyEqual(self, left, right, epsilon = 0.05):
        if type(left)  is list: left  = np.array(left)
        if type(right) is list: right = np.array(right)

        return self.assertTrue(np.allclose(left, right))

class TestGeometry(TestNumpyArray):
    def setUp(self):
        self.point = np.array([1,0,0,1])
        self.points = np.array([[1,0,0,1],
                                [0,1,0,1],
                                [0,0,1,1]])

    def test_point_translation(self):
        translated_point = np.dot(translation_matrix(np.array([1,1,1])), self.point)

        self.assertArraysEqual(translated_point, [2,1,1,1])

    def test_cloud_translation(self):
        translated_points = apply_transform(translation_matrix(np.array([1,1,1])), self.points)

        self.assertArraysEqual(translated_points, [[2,1,1,1],
                                                   [1,2,1,1],
                                                   [1,1,2,1]])

    def test_point_rotation(self):
        rotated_point = np.dot(promote(rotation_matrix(0,0,math.pi/2)), self.point)

        self.assertArraysApproximatelyEqual(rotated_point, [0.,1.,0.,1.])

    def test_cloud_rotation(self):
        rotated_points = apply_transform(promote(rotation_matrix(0,0,math.pi/2)), self.points)

        self.assertArraysApproximatelyEqual(rotated_points, [[0.,1.,0.,1.],
                                                             [-1,0.,0.,1.],
                                                             [0.,0.,1.,1.]])

class TestNNSHelpers(TestNumpyArray):
    def setUp(self):
        self.triangle = np.array([[0,0,0,1],
                                  [1,0,0,1],
                                  [1,1,0,1]])

    def test_covariance_of_self_is_zero(self):
        self.assertArraysApproximatelyEqual(cross_covariance(self.triangle, self.triangle), np.zeros((4,4)))

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
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGeometry)
    suite.addTest(TestNNSHelpers("test_covariance_of_self_is_zero"))
    unittest.TextTestRunner(verbosity=2).run(suite)
