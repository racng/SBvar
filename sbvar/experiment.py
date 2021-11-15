import tellurium as te
from roadrunner import RoadRunner

class Experiment(
):
    """
    Performs robustness analysis by performing simulations for a range of 
    parameter values.

    Parameters
    ----------
    rr : `RoadRunner` object
        RoadRunner object for simulating SBML model.
    start: float (default:0)
        Starting time for simulations
    end: float (default:5)
        Ending time for simulations
    points: int (default: 51)
        Number of time points to return for each simulation.
    selections: list of str
        A list of variables to be recorded in simulation output. 
    steps:
    """
    def __init__(self, rr, start=0, end=5, points=51, 
        selections=None, steps=None):
        self.rr = rr
        self.start = start
        self.end = end
        self.points

