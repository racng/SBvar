import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sbvar import utils

import tellurium as te

from sbvar.experiment import OneWayExperiment

ant_uni = '''
    J0: S1 -> S2; k1*S1;
    J1: S2 -> S3; k2*S2;

    k1= 0.1; k2 = 0.2;
    S1 = 10; S2 = 0; S3 = 0;
'''

ant_bi = '''
    J0: $Xo -> S1; 1 + Xo*(32+(S1/0.75)^3.2)/(1 +(S1/4.3)^3.2);
    J1: S1 -> $X1; k1*S1;

    Xo = 0.09; X1 = 0.0;
    S1 = 0.5; k1 = 3.2;
'''

class TestOneWayExperiment(unittest.TestCase):
    def setUp(self) -> None:
        self.sim_kwargs = {'start':0, 'end':40, 'points':100, 'steps':None}
        self.param_kwargs = {'param':'S1', 'bounds':(0, 10), 'num':5}
        self.rr = te.loada(ant_bi)
        self.exp = OneWayExperiment(self.rr, selections=None, 
            conserved_moiety=False, **self.sim_kwargs, **self.param_kwargs)
    
    def test_init(self):
        self.assertCountEqual(self.exp.species_ids, ['S1'])
        self.assertCountEqual(self.exp.boundary_ids, ["Xo", "X1"])
        self.assertCountEqual(self.exp.flux_ids, ["S1'"])
        self.assertCountEqual(self.exp.reaction_ids, ["J0", "J1"])
        self.assertEqual(self.exp.dim, 1)
        self.assertEqual(self.exp.param, self.param_kwargs['param'])
        self.assertEqual(self.exp.bounds, self.param_kwargs['bounds'])
        self.assertEqual(self.exp.num, self.param_kwargs['num'])

        self.assertEqual(self.exp.simulations, None)
        self.assertEqual(self.exp.steady_states, None)
    
    def test_set_conditions(self):
        self.param_kwargs = {'param':'S1', 'bounds':(0, 10), 'num':5}
        self.exp = OneWayExperiment(self.rr, **self.sim_kwargs, **self.param_kwargs)
        self.assertTrue(np.allclose(self.exp.conditions, [0, 2.5, 5, 7.5, 10]))
        
        self.param_kwargs = {'param':'S1', 'levels':[0, 1, 2, 3]}
        self.exp = OneWayExperiment(self.rr, **self.sim_kwargs, **self.param_kwargs)
        self.assertTrue(np.allclose(self.exp.conditions, [0, 1, 2, 3]))

        self.param_kwargs = {'param':'S1', 'levels':5}
        self.assertRaises(TypeError, OneWayExperiment, self.rr, 
            **self.sim_kwargs, **self.param_kwargs)

        self.param_kwargs = {'param':'S1', 'bounds':5, 'num':5}
        self.assertRaises(TypeError, OneWayExperiment, self.rr, 
            **self.sim_kwargs, **self.param_kwargs)

        self.param_kwargs = {'param':'S1', 'bounds':(0, 10, 50), 'num':5}
        self.assertRaises(ValueError, OneWayExperiment, self.rr, 
            **self.sim_kwargs, **self.param_kwargs)

        self.param_kwargs = {'param':'S1', 'bounds':(0, 'a'), 'num':5}
        self.assertRaises(TypeError, OneWayExperiment, self.rr, 
            **self.sim_kwargs, **self.param_kwargs)

    def test_get_conditions_df(self):
        expected = pd.DataFrame({'S1':[0, 2.5, 5, 7.5, 10]})
        pd.testing.assert_frame_equal(expected, self.exp.get_conditions_df())

    def test_iter_conditions(self):
        def save_value():
            return self.exp.rr[self.param_kwargs['param']]
        values = self.exp.iter_conditions(save_value)
        self.assertCountEqual(values, [0, 2.5, 5, 7.5, 10])

    def test_conditions_to_meshes(self):
        start, end = self.sim_kwargs['start'], self.sim_kwargs['end']
        points = self.sim_kwargs['points']
        t = np.linspace(start, end, points)
        y = [0, 2.5, 5, 7.5, 10]
        expected = np.meshgrid(t, y)
        meshes = self.exp.conditions_to_meshes()
        np.testing.assert_array_equal(meshes, expected)

    def test_get_timecourese_mesh(self):
        start, end = self.sim_kwargs['start'], self.sim_kwargs['end']
        points = self.sim_kwargs['points']
        t = np.linspace(start, end, points)
        t_final = []
        for x in [0, 2.5, 5, 7.5, 10]:
            self.rr.reset()
            self.rr['S1'] = x
            out = self.rr.simulate(**self.sim_kwargs, selections=['S1'])
            t_final.append(out.flatten()[-1])

        Z = self.exp.get_timecourse_mesh('S1')
        np.testing.assert_allclose(Z[:, 0], [0, 2.5, 5, 7.5, 10])
        np.testing.assert_allclose(Z[:, -1], t_final)

    def test_plot_timecourse_mesh(self):
        self.exp.simulate()
        combos = [
            ('contourf', '2d'), ('contour', '2d'), 
            ('contourf', '3d'), ('contour', '3d'),
            ('surface', '3d')]
        for kind, proj in combos:
            self.exp.plot_timecourse_mesh('S1', kind=kind, projection=proj)
            plt.close()
        self.assertRaises(ValueError, self.exp.plot_timecourse_mesh, 'S1',
            kind='surface', projection='2d')

if __name__ == '__main__':
    unittest.main()
