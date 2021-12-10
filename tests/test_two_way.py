import unittest
import numpy as np
import pandas as pd
from sbvar import utils

import tellurium as te

from sbvar.experiment import TwoWayExperiment

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

class TestTwoWayExperiment(unittest.TestCase):
    def setUp(self) -> None:
        self.sim_kwargs = {'start':0, 'end':40, 'points':100, 'steps':None}
        self.param_kwargs = {
            'param1':'S1', 'param2':'Xo', 
            'bounds1':(0, 10), 'num1':5,
            'bounds2':(0.8, 1), 'num2':3,
            }
        self.rr = te.loada(ant_bi)
        self.exp = TwoWayExperiment(self.rr, selections=None, 
            conserved_moiety=False, **self.sim_kwargs, **self.param_kwargs)
    
    def test_init(self):
        self.assertCountEqual(self.exp.species_ids, ['S1'])
        self.assertCountEqual(self.exp.boundary_ids, ["Xo", "X1"])
        self.assertCountEqual(self.exp.flux_ids, ["S1'"])
        self.assertCountEqual(self.exp.reaction_ids, ["J0", "J1"])
        self.assertEqual(self.exp.dim, 2)
        self.assertEqual(self.exp.param_list, ['S1', 'Xo'])
        self.assertEqual(self.exp.bounds_list, 
            [self.param_kwargs['bounds1'], self.param_kwargs['bounds2']])
        self.assertEqual(self.exp.levels_list, 
            [None, None])
        self.assertEqual(self.exp.simulations, None)
        self.assertEqual(self.exp.steady_states, None)

    def test_set_conditions(self):
        self.param_kwargs = {
            'param1':'S1', 'param2':'Xo', 
            'bounds1':(0, 10), 'num1':5,
            'bounds2':(0.8, 1), 'num2':3,
            }
        expected = np.array(
            [[ 0. ,  0.8],
            [ 2.5,  0.8],
            [ 5. ,  0.8],
            [ 7.5,  0.8],
            [10. ,  0.8],
            [ 0. ,  0.9],
            [ 2.5,  0.9],
            [ 5. ,  0.9],
            [ 7.5,  0.9],
            [10. ,  0.9],
            [ 0. ,  1. ],
            [ 2.5,  1. ],
            [ 5. ,  1. ],
            [ 7.5,  1. ],
            [10. ,  1. ]])
        self.exp = TwoWayExperiment(self.rr, **self.sim_kwargs, **self.param_kwargs)
        self.assertTrue(np.allclose(self.exp.conditions, expected))
        np.testing.assert_equal(self.exp.conditions_list, 
            [[0, 2.5, 5, 7.5, 10], [0.8, 0.9, 1.]])

        self.param_kwargs = {
            'param1':'S1', 'param2':'Xo', 
            'bounds1':(0, 10), 'num1':5,
            'levels2': [0.89, 0.9, 0.91]
            }
        expected = np.array(
            [[ 0.  ,  0.89],
            [ 2.5 ,  0.89],
            [ 5.  ,  0.89],
            [ 7.5 ,  0.89],
            [10.  ,  0.89],
            [ 0.  ,  0.9 ],
            [ 2.5 ,  0.9 ],
            [ 5.  ,  0.9 ],
            [ 7.5 ,  0.9 ],
            [10.  ,  0.9 ],
            [ 0.  ,  0.91],
            [ 2.5 ,  0.91],
            [ 5.  ,  0.91],
            [ 7.5 ,  0.91],
            [10.  ,  0.91]])
        self.exp = TwoWayExperiment(self.rr, **self.sim_kwargs, **self.param_kwargs)
        self.assertTrue(np.allclose(self.exp.conditions, expected))
        np.testing.assert_equal(self.exp.conditions_list, 
            [[0, 2.5, 5, 7.5, 10], [0.89, 0.9, 0.91]])

        self.param_kwargs = {
            'param1':'S1', 'param2':'Xo', 
            'bounds1':(0, 10), 'num1':5,
            'levels2': 5
            }
        self.assertRaises(TypeError, TwoWayExperiment, self.rr, 
            **self.sim_kwargs, **self.param_kwargs)

        self.param_kwargs = {
            'param1':'S1', 'param2':'Xo', 
            'bounds1':5, 'num1':5,
            'bounds2':(0.8, 1), 'num2':3,
            }
        self.assertRaises(TypeError, TwoWayExperiment, self.rr, 
            **self.sim_kwargs, **self.param_kwargs)

        self.param_kwargs = {
            'param1':'S1', 'param2':'Xo', 
            'bounds1':(0, 10, 15), 'num1':5,
            'bounds2':(0.8, 1), 'num2':3,
            }
        self.assertRaises(ValueError, TwoWayExperiment, self.rr, 
            **self.sim_kwargs, **self.param_kwargs)

        self.param_kwargs = {
            'param1':'S1', 'param2':'Xo', 
            'bounds1':(0, 'a'), 'num1':5,
            'bounds2':(0.8, 1), 'num2':3,
            }
        self.param_kwargs = {'param':'S1', 'bounds':(0, 'a'), 'num':5}
        self.assertRaises(TypeError, TwoWayExperiment, self.rr, 
            **self.sim_kwargs, **self.param_kwargs)

    
    
if __name__ == '__main__':
    unittest.main()
