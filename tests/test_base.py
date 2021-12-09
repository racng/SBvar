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
        self.rr = te.loada(ant_uni)
        self.exp = Experiment(self.rr, start=0, end=5, points=51, 
            steps=None, selections=None)

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
        expected = ['time', 'S1', 'S2', 'S3', "S1'", "S2'", "S3'", 'J0', 'J1']
        exp.set_selections(None)
        self.assertCountEqual(exp.selections, expected)

        # Custom selections
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
        self.assertEqual(self.exp.iter_conditions(dummy), "dummy")

    def test_simulate(self):
        self.exp.simulate()
        return

    