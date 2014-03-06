#!/usr/bin/env python

from objfile import *
from sys import argv
import numpy as np
from nns import *
import unittest

class TestNumpyArray(unittest.TestCase):
    def assertArraysEqual(self, left, right, msg = None):
        if type(left)  is list: left  = np.array(left)
        if type(right) is list: right = np.array(right)

        return self.assertTrue(np.array_equal(left, right), msg)

    def assertArraysApproximatelyEqual(self, left, right, msg = None, epsilon = 1e-05):
        if type(left)  is list: left  = np.array(left)
        if type(right) is list: right = np.array(right)

        return self.assertTrue(np.allclose(left, right, epsilon), msg)

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
        self.nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(self.triangle)

    def test_covariance(self):
        self.assertArraysApproximatelyEqual(
                cross_covariance(self.triangle, self.triangle), [[2/9., 1/9., 0, 0],
                                                                 [1/9., 2/9., 0, 0],
                                                                 [0, 0, 0, 0],
                                                                 [0, 0, 0, 0]])
    def test_closest_point(self):
        test_point = np.array([2,2,0,1])
        closest_point_index = self.nn.kneighbors(test_point)[1]

        self.assertArraysApproximatelyEqual(self.triangle[closest_point_index], [1,1,0,1])

        test_point = np.array([-1,-2,0,1])
        closest_point_index = self.nn.kneighbors(test_point)[1]

        self.assertArraysApproximatelyEqual(self.triangle[closest_point_index], [0,0,0,1])

    def test_mean_square_error_of_self_is_zero(self):
        self.assertEqual(mean_square_error(self.triangle, self.triangle), 0)

class TestNNS(TestNumpyArray):
    def setUp(self):
        self.triangle = np.array([[0,0,0,1],
                                  [1,0,0,1],
                                  [0,1,0,1]])

        self.stuff = load_obj_file("./obj/shorts1vertex70.obj")
        self.large = self.stuff[1]

    def test_icp_under_identity(self):
        result = icp(self.triangle, self.triangle.copy())

        print
        print result

        self.assertArraysApproximatelyEqual(result, np.identity(4), 
                "Result should be close to the 4x4 identity matrix.")

    def test_large_icp_under_identity(self):
        result = icp(self.large, self.large.copy())

        print
        print result

        self.assertArraysApproximatelyEqual(result, np.identity(4), 
                "Result should be close to the 4x4 identity matrix.")


    def test_icp_under_rotation(self):
        self.test_cloud_under_transformation(self.triangle, np.array([0,0,0,0]), [0,0,math.pi/4])

    def test_icp_under_translation(self):
        self.test_cloud_under_transformation(self.triangle, np.array([.9,1,0,0]), [0.9,0,0])


    def test_large_icp_under_rotation(self):
        transform = promote(rotation_matrix(math.pi/9,math.pi/4,math.pi/9))
        transformed = apply_transform(transform, self.large.copy())
        result = icp(self.large, transformed)

        print
        print "Actual:"
        print transform
        print "Predicted:"
        print result

        save_obj_file("./obj/shorts1vertex70transformed.obj", self.stuff[0], transformed, self.stuff[2])
        save_obj_file("./obj/shorts1vertex70predicted.obj", self.stuff[0], apply_transform(result.T, self.large.copy()), self.stuff[2])
        self.assertArraysApproximatelyEqual(result.T, transform)
    def test_large_icp_under_rotation_and_translation(self):
        self.test_cloud_under_transformation(self.large, np.array([.9,1,0,0]), [0.9,0,0])

    def test_cloud_under_transformation(self, cloud, translation, rotation):
        transform = np.dot(promote(rotation_matrix(rotation[0], rotation[1], rotation[2])), translation_matrix(translation))


        moved_cloud = apply_transform(transform, cloud.copy())
        print "Original point cloud:"
        print cloud
        print "Actual point cloud:"
        print moved_cloud
        print "Actual Transform:"
        print transform

        result = icp(moved_cloud, cloud)
        print "Predicted Transform:"
        print result
        print "Predicted point cloud:"
        print apply_transform(result, cloud.copy())

        self.assertArraysApproximatelyEqual(result, transform)
        
if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestGeometry)

    suite.addTest(TestNNSHelpers("test_covariance"))
    suite.addTest(TestNNSHelpers("test_closest_point"))
    suite.addTest(TestNNSHelpers("test_mean_square_error_of_self_is_zero"))

    #suite.addTest(TestNNS("test_icp_under_identity"))
    #suite.addTest(TestNNS("test_large_icp_under_identity"))
    suite.addTest(TestNNS("test_icp_under_rotation"))
    #suite.addTest(TestNNS("test_icp_under_translation"))
    suite.addTest(TestNNS("test_large_icp_under_rotation"))
    suite.addTest(TestNNS("test_large_icp_under_rotation_and_translation"))

    unittest.TextTestRunner(verbosity=2).run(suite)
