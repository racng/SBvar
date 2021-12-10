import unittest
import numpy as np
from numpy.lib.function_base import select
from sbvar import utils

import tellurium as te

from sbvar.experiment import Experiment

ant_uni = '''
    J0: S1 -> S2; k1*S1;
    J1: S2 -> S3; k2*S2;

    k1= 0.1; k2 = 0.2;
    S1 = 10; S2 = 0; S3 = 0;
'''
ant_bi = '''
    $Xo -> S1; 1 + Xo*(32+(S1/0.75)^3.2)/(1 +(S1/4.3)^3.2);
    S1 -> $X1; k1*S1;

    Xo = 0.09; X1 = 0.0;
    S1 = 0.5; k1 = 3.2;
'''

class TestExperiment(unittest.TestCase):
    def setUp(self) -> None:
        self.kwargs = {'start':0, 'end':500, 'points':100, 'steps':None}
        self.selections = ['time', 'S1', 'S2', 'S3', "S1'", "S2'", "S3'", 'J0', 'J1']
        self.rr = te.loada(ant_uni)
        self.exp = Experiment(self.rr, selections=None, conserved_moiety=False, 
            **self.kwargs)

    def test_init(self):
        exp = self.exp
        self.assertCountEqual(exp.species_ids, ['S1', 'S2', 'S3'])
        self.assertCountEqual(exp.boundary_ids, [])
        self.assertCountEqual(exp.flux_ids, ["S1'", "S2'", "S3'"])
        self.assertCountEqual(exp.reaction_ids, ["J0", "J1"])
        self.assertEqual(exp.dim, 0)
        self.assertEqual(exp.conditions, None)
        self.assertEqual(exp.simulations, None)
        self.assertEqual(exp.steady_states, None)

    def test_check_in_rr(self):
        exp = self.exp
        exp.check_in_rr("S1") # no error raised
        self.assertRaises(ValueError, exp.check_in_rr, "S4")

    def test_set_selections(self):
        # Default behavior
        exp = self.exp
        expected = self.selections
        exp.set_selections(None)
        self.assertCountEqual(exp.selections, expected)

        # Custom selections
        self.assertRaises(TypeError, exp.set_selections, "S1")
        exp.set_selections(['S1', 'S2', 'S3'])
        self.assertWarns(UserWarning, exp.set_selections, ['S1', 'S2', 'S3'])
        expected = ['time', 'S1', 'S2', 'S3']
        self.assertCountEqual(exp.selections, expected)

    def test_set_steady_state_selections(self):
        # Default behavior
        exp=self.exp

        expected = ['S1', 'S2', 'S3', 'J0', 'J1']
        exp.set_steady_state_selections(None)
        self.assertCountEqual(exp.steady_state_selections, expected)

        # Custom selections
        expected = ['S1', 'S2', 'S3']
        exp.set_steady_state_selections(['S1', 'S2', 'S3'])
        self.assertCountEqual(exp.steady_state_selections, expected)
        return

    def test_iter_conditions(self):
        def dummy():
            return 'dummy'
        expected = ['dummy']
        self.assertEqual(self.exp.iter_conditions(dummy), expected)

    def test_simulate(self):
        out = self.rr.simulate(selections=self.selections, **self.kwargs)
        expected = np.dstack([out])
        self.exp.simulate()
        self.assertTrue(np.array_equal(self.exp.simulations, expected))

        self.rr.reset()
        self.exp.conserved_moiety = True
        self.exp.simulate()
        self.assertTrue(np.array_equal(self.exp.simulations, expected))
        return

    def test_calc_steady_state(self):
        # Check warning if conservedMoietyAnalysis is False
        self.assertWarns(UserWarning, self.exp.calc_steady_state)

        # Check steady state if conservedMoietyAnalysis is True
        self.rr.conservedMoietyAnalysis = True
        self.rr.steadyStateSelections = ['S1', 'S2', 'S3', 'J0', 'J1']
        out = self.rr.getSteadyStateValues()
        expected = np.vstack([out])

        self.exp.conserved_moiety = True
        self.exp.calc_steady_state()
        self.assertTrue(np.allclose(self.exp.steady_states, expected))
        self.assertAlmostEqual(self.exp.steady_states[0, 0], 0)
        self.assertAlmostEqual(self.exp.steady_states[0, 2], 10)

    def test_get_selection_index(self):
        self.assertEqual(self.exp.get_selection_index("time"), 0)
        self.assertEqual(self.exp.get_selection_index("S1"), 1)
        self.assertRaises(ValueError, self.exp.get_selection_index, "S4")


    def test_get_steady_state(self):
        self.assertRaises(ValueError, self.exp.get_steady_state, "S4")

        self.rr.conservedMoietyAnalysis = True
        self.rr.steadyStateSelections = ['S1', 'S2', 'S3', 'J0', 'J1']
        ss = np.vstack([self.rr.getSteadyStateValues()])
        
        self.exp.conserved_moiety = True
        self.assertWarns(UserWarning, self.exp.get_steady_state, "S1")
        self.exp.calc_steady_state()
        self.assertTrue(np.allclose(self.exp.get_steady_state('S1'), ss[:, 0]))
        self.assertTrue(np.allclose(self.exp.get_steady_state('S2'), ss[:, 1]))
        self.assertTrue(np.allclose(self.exp.get_steady_state('S3'), ss[:, 2]))
        self.assertTrue(np.allclose(self.exp.get_steady_state('J0'), ss[:, 3]))
        self.assertTrue(np.allclose(self.exp.get_steady_state('J1'), ss[:, 4]))
        
    def test_get_step_values(self):
        self.assertWarns(UserWarning, self.exp.get_step_values, "S1", 0)
        self.exp.simulate()
        # Check initial values
        self.assertTrue(np.allclose(self.exp.get_step_values('S1', 0), [10]))
        self.assertTrue(np.allclose(self.exp.get_step_values('S2', 0), [0]))
        self.assertTrue(np.allclose(self.exp.get_step_values('S3', 0), [0]))
        # Check approximate steady state values
        self.assertTrue(np.allclose(self.exp.get_step_values('S3', -1), [10]))
        self.assertTrue(np.allclose(self.exp.get_step_values('S2', -1), [0]))
        self.assertTrue(np.allclose(self.exp.get_step_values('S1', -1), [0]))

    def test_get_timepoints(self):
        self.assertWarns(UserWarning, self.exp.get_timepoints)
        
        expected = np.linspace(self.kwargs['start'], self.kwargs['end'], 
            num=self.kwargs['points'])
        self.assertTrue(np.allclose(self.exp.get_timepoints(), expected))

    def test_get_closest_timepoint(self):
        self.assertEqual(self.exp.get_closest_timepoint(0), 0)
        end, points = self.kwargs['end'], self.kwargs['points']
        self.assertEqual(self.exp.get_closest_timepoint(end), points - 1)

    def test_get_time_values(self):
        self.assertWarns(UserWarning, self.exp.get_time_values, "S1", 0)
        self.exp.simulate()
        # Check initial values
        self.assertTrue(np.allclose(self.exp.get_time_values('S1', 0), [10]))
        self.assertTrue(np.allclose(self.exp.get_time_values('S2', 0), [0]))
        self.assertTrue(np.allclose(self.exp.get_time_values('S3', 0), [0]))
        # Check approximate steady state values
        self.assertTrue(np.allclose(self.exp.get_time_values('S3', 500), [10]))
        self.assertTrue(np.allclose(self.exp.get_time_values('S2', 500), [0]))
        self.assertTrue(np.allclose(self.exp.get_time_values('S1', 500), [0]))

    def test_get_values(self):
        self.assertRaises(ValueError, self.exp.get_values, 'S4')
        self.assertRaises(ValueError, self.exp.get_values, 'S1', 
            steady_state=True, time=1)
        self.assertRaises(ValueError, self.exp.get_values, 'S1', 
            steady_state=True, time=1)
        self.assertRaises(ValueError, self.exp.get_values, 'S1', 
            step=1, time=1)

        self.exp.conserved_moiety = True
        expected = self.exp.get_steady_state('S1')
        self.assertCountEqual(self.exp.get_values('S1'), expected)

        expected = self.exp.get_step_values('S1', -1)
        self.assertCountEqual(
            self.exp.get_values('S1', steady_state=False, step=-1), 
            expected)

        expected = self.exp.get_time_values('S1', 500)
        self.assertCountEqual(
            self.exp.get_values('S1', steady_state=False, time=500), 
            expected)

if __name__ == '__main__':
    unittest.main()
        
        


        

