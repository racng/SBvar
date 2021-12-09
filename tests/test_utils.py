import unittest
import numpy as np
from sbvar import utils

class TestMeshConverters(unittest.TestCase):

    def setUp(self):
        return 

    def test_meshes_to_meshvector(self):
        x = np.arange(0, 2, 1)
        y = np.arange(4, 7, 1)
        meshes = np.meshgrid(x, y)
        meshvector = utils.meshes_to_meshvector(meshes)
        expect = np.array(
            [[0, 4],
            [1, 4],
            [0, 5],
            [1, 5],
            [0, 6],
            [1, 6]])
        self.assertTrue(np.array_equal(meshvector, expect))

    def test_vector_to_mesh(self):
        x = np.array([1, 2, 3, 4, 5, 6])
        mesh = utils.vector_to_mesh(x, 2, 3)
        truth = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(np.array_equal(mesh, truth))

    def test_meshvector_to_meshes(self):
        meshvector = np.array(
            [[0, 4],
            [1, 4],
            [0, 5],
            [1, 5],
            [0, 6],
            [1, 6]])
        X, Y = utils.meshvector_to_meshes(meshvector, 2, 3)
        X_truth = np.array([[0, 1], [0, 1], [0, 1]])
        Y_truth = np.array([[4, 4], [5, 5], [6, 6]])
        self.assertTrue(np.array_equal(X, X_truth))
        self.assertTrue(np.array_equal(Y, Y_truth))

if __name__ == '__main__':
    unittest.main()