#.!/usr/bin/env python

from ..objfile import *
from ..identify import *

from sys import argv
import numpy as np
import unittest

class TestNumpyArray(unittest.TestCase):
    def assert_arrays_equal(self, left, right, msg = None):
        if type(left)  is list: left  = np.array(left)
        if type(right) is list: right = np.array(right)

        return self.assertTrue(np.array_equal(left, right), msg)

    def assert_array_approximately_equal(self, left, right, msg = None, epsilon = 1e-05):
        if type(left)  is list: left  = np.array(left)
        if type(right) is list: right = np.array(right)

        return self.assertTrue(np.allclose(left, right, atol = epsilon), msg)

class TestGeometry(TestNumpyArray):
    def setUp(self):
        self.point = np.array([1,0,0,1])
        self.points = np.array([[1,0,0,1],
                                [0,1,0,1],
                                [0,0,1,1]])

    def test_point_translation(self):
        translated_point = np.dot(translation_matrix(np.array([1,1,1])), self.point)

        self.assert_arrays_equal(translated_point, [2,1,1,1])

    def test_cloud_translation(self):
        translated_points = apply_transform(translation_matrix(np.array([1,1,1])), self.points)

        self.assert_arrays_equal(translated_points, [[2,1,1,1],
                                                   [1,2,1,1],
                                                   [1,1,2,1]])

    def test_point_rotation(self):
        rotated_point = np.dot(promote(rotation_matrix(0,0,math.pi/2)), self.point)

        self.assert_array_approximately_equal(rotated_point, [0.,1.,0.,1.])

    def test_cloud_rotation(self):
        rotated_points = apply_transform(promote(rotation_matrix(0,0,math.pi/2)), self.points)

        self.assert_array_approximately_equal(rotated_points, [[0.,1.,0.,1.],
                                                             [-1,0.,0.,1.],
                                                             [0.,0.,1.,1.]])

    def test_arbitrary_axis_rotation_at_arbitrary_origin(self):
        import random
        theta = random.random()

        # does arbitrary_axis_rotation work?
        self.assert_array_approximately_equal(
                arbitrary_axis_rotation(np.array([1,0,0]), theta),
                promote(rotation_matrix(theta,0,0))
                    )


        # the origin should NOT be rotated
        pt = np.array([1,0,0])
        mat = arbitrary_axis_rotation_at_arbitrary_origin(np.array([1,0,0]), pt, theta)
        pt = vector_to_affine(pt)

        self.assert_array_approximately_equal(np.dot(mat, pt), pt)

    def test_unify_segments_matrix(self):
        import random 

        NUM_TESTS = 100
        for i in range(NUM_TESTS):
            MAX_DISTANCE = 10

            def affine(v):
                result = np.ones(4)
                result[0:3] = v
                return result

            origin   = np.random.rand(3)
            distance = random.random() * MAX_DISTANCE + 1

            a = (unit_vector(np.random.rand(3)) * distance) + origin
            c = (unit_vector(np.random.rand(3)) * distance) + origin

            mat = get_unify_segments_matrix(a,origin,c)

            a = affine(a)
            c = affine(c)

            self.assert_array_approximately_equal(a, np.dot(mat, c), "", 0.1)

    def test_principal_axes(self):
        array = np.array([[1,0,0], [2,0,0], [3,0,0],
                          [0,0.1,0], [0,0.2,0], [0,0.3,0],
                          [0,0,1], [0,0,2], [0,0,3],
                          ])

        axis = principal_axis(array)
        expected = np.array([1,0,0])

        #self.assert_array_approximately_equal(axis, expected, epsilon = 1e-1)

        print principal_axes(array)

class TestNNSHelpers(TestNumpyArray):
    def setUp(self):
        self.triangle = np.array([[0,0,0,1],
                                  [1,0,0,1],
                                  [1,1,0,1]])
        self.nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(self.triangle)

    def test_covariance(self):
        self.assert_array_approximately_equal(
                cross_covariance(self.triangle, self.triangle), [[2/9., 1/9., 0, 0],
                                                                 [1/9., 2/9., 0, 0],
                                                                 [0, 0, 0, 0],
                                                                 [0, 0, 0, 0]])
    def test_closest_point(self):
        test_point = np.array([2,2,0,1])
        closest_point_index = self.nn.kneighbors(test_point)[1]

        self.assert_array_approximately_equal(self.triangle[closest_point_index], [1,1,0,1])

        test_point = np.array([-1,-2,0,1])
        closest_point_index = self.nn.kneighbors(test_point)[1]

        self.assert_array_approximately_equal(self.triangle[closest_point_index], [0,0,0,1])

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
        result = icp(self.triangle, self.triangle.copy())[0]

        print
        print result

        self.assert_array_approximately_equal(result, np.identity(4), 
                "Result should be close to the 4x4 identity matrix.")

    def test_large_icp_under_identity(self):
        result = icp(self.large, self.large.copy())[0]

        print
        print result

        self.assert_array_approximately_equal(result, np.identity(4), 
                "Result should be close to the 4x4 identity matrix.")


    def test_icp_under_rotation(self):
        self.test_cloud_under_transformation(self.triangle, np.array([0,0,0,0]), [0,0,math.pi/4])

    def test_icp_under_translation(self):
        self.test_cloud_under_transformation(self.triangle, np.array([.9,1,0,0]), [0.9,0,0])


    def test_large_icp_under_rotation(self):
        transform = promote(rotation_matrix(math.pi/9,math.pi/4,math.pi/9))
        transformed = apply_transform(transform, self.large.copy())
        result = icp(self.large, transformed)[0]

        print
        print "Actual:"
        print transform
        print "Predicted:"
        print result

        # save_obj_file("./obj/shorts1vertex70transformed.obj", self.stuff[0], transformed, self.stuff[2])
        # save_obj_file("./obj/shorts1vertex70predicted.obj", self.stuff[0], apply_transform(result.T, self.large.copy()), self.stuff[2])
        self.assert_array_approximately_equal(result.T, transform)

    def test_large_icp_under_rotation_and_translation(self):
        self.test_cloud_under_transformation(self.large, np.array([1,1,1,0]), [0.9,1,0])

    def test_cloud_under_transformation(self, cloud, translation, rotation):
        transform = np.dot(promote(rotation_matrix(rotation[0], rotation[1], rotation[2])), translation_matrix(translation))


        moved_cloud = apply_transform(transform, cloud.copy())
        print "Original point cloud:"
        print cloud
        print "Actual point cloud:"
        print moved_cloud
        print "Actual Transform:"
        print transform

        result = icp(moved_cloud, cloud)[0]
        print "Predicted Transform:"
        print result
        print "Predicted point cloud:"
        print apply_transform(result, cloud.copy())

        self.assert_array_approximately_equal(result, transform)

TEST_TRIMESH = "./nns/testdata/40.obj"
class TrimeshTest(TestNumpyArray):
    def setUp(self):
        self.mesh = TriMesh.FromOBJ_FileName(TEST_TRIMESH)
        pass

    def test_trimesh_transform_function(self):
        test = self.mesh.copy()

        translate = np.array([1,0,0])
        transform_trimesh(test, lambda p: p + translate)

        M = translation_matrix(translate)
        affine_transform_trimesh(test, M)

        self.assert_array_approximately_equal(test.vs[0], self.mesh.vs[0] + 2*translate)

        
if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    VERBOSE  = False
    DO_SCALE = True

    suite = unittest.TestLoader().loadTestsFromTestCase(TestGeometry)

    suite.addTest(TestNNSHelpers("test_covariance"))
    suite.addTest(TestNNSHelpers("test_closest_point"))
    suite.addTest(TestNNSHelpers("test_mean_square_error_of_self_is_zero"))

    #suite.addTest(TestNNS("test_icp_under_identity"))
    #suite.addTest(TestNNS("test_large_icp_under_identity"))
    #suite.addTest(TestNNS("test_icp_under_rotation"))
    #suite.addTest(TestNNS("test_icp_under_translation"))
    #suite.addTest(TestNNS("test_large_icp_under_rotation"))
    #suite.addTest(TestNNS("test_large_icp_under_rotation_and_translation"))

    #suite.addTest(TrimeshTest("test_trimesh_transform_function"))

    unittest.TextTestRunner(verbosity=2).run(suite)
