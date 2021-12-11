import unittest
import numpy as np
from numpy.lib.function_base import select
from sbvar import utils

import tellurium as te

from sbvar.experiment import OneWayExperiment
from sbvar.timeseries import *
ant_bi = '''
    J0: $Xo -> S1; 1 + Xo*(32+(S1/0.75)^3.2)/(1 +(S1/4.3)^3.2);
    J1: S1 -> $X1; k1*S1;

    Xo = 0.09; X1 = 0.0;
    S1 = 0.5; k1 = 3.2;
'''

class TestTimeseries(unittest.TestCase):
    def setUp(self):
        self.sim_kwargs = {'start':0, 'end':40, 'points':100, 'steps':None}
        self.param_kwargs = {'param':'S1', 'bounds':(0, 12), 'num':40}
        self.rr = te.loada(ant_bi)
        self.exp = OneWayExperiment(self.rr, selections=None, 
            conserved_moiety=False, **self.sim_kwargs, **self.param_kwargs)
        
    def test_cluster_timeseries(self):
        self.exp.simulate()
        labels = cluster_timeseries(self.exp, 'S1', 'kmeans', n_clusters=4)
        np.testing.assert_allclose(labels, self.exp.obs['kmeans_S1'].values)
        self.assertEqual(set(labels), {0,1,2,3})

    def test_kmeans(self):
        X = [[1, 0, 0],
            [1, 0, 0], 
            [0, 0, 1]]
        labels = kmeans(X, n_clusters=2)
        self.assertTrue(labels[0]==labels[1])
        self.assertTrue(labels[0]!=labels[2])

    def plot_timecourse_clusters(self):
        self.exp.simulate()
        labels = cluster_timeseries(self.exp, 'S1', 'kmeans', n_clusters=4)
        plot_timecourse_clusters(self.exp, 'S1', 'kmeans')
        
    