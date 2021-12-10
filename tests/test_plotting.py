import unittest
import numpy as np
import pandas as pd
from sbvar.plotting import *

import tellurium as te

def sum_squares(x, y):
    return x**2 + y**2

class TestPlotting(unittest.TestCase):
    def setUp(self):
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = sum_squares(self.X, self.Y)

    def test_plot_mesh(self):
        combos = [
            ('contourf', '2d'), ('contour', '2d'), 
            ('contourf', '3d'), ('contour', '3d'),
            ('surface', '3d')]
        for kind, proj in combos:
            plot_mesh(self.X, self.Y, self.Z, kind=kind, projection=proj)

        self.assertRaises(ValueError, plot_mesh, self.X, self.Y, self.Z, 
            kind='surface', projection='2d')

if __name__ == '__main__':
    unittest.main()