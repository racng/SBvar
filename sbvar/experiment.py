import numpy as np
import tellurium as te
from roadrunner import RoadRunner
import warnings

doc_simulation = """
    rr : `RoadRunner` object
        RoadRunner object for simulating SBML model.
    start: float (default:0)
        Starting time for simulations
    end: float (default:5)
        Ending time for simulations
    points: int (default: 51)
        Number of time points to return for each simulation.
    steps: int (optional)
        Number of steps at which output is sampled, 
        where the samples are evenly spaced.
        Steps = points - 1. Steps and points may not both be specified. 
    selections: list of str (optional, default: None)
        A list of variables to be recorded in simulation output. 
        Time will be added to selections if not specified. If None, 
        keeps track of time and all floating species'amounts and derivatives 
        and reaction rates.
"""

# @_doc_params(simulation_params=doc_simulation)
class Experiment():
    """
    Performs robustness analysis by performing simulations for a range of 
    parameter values.

    Parameters
    ----------
    {simulation_params}

    Attributes
    ----------
    conditions: list of tuples
        List of levels 
    simulations: list of NamedArray
        List of simulation output
    """
    def __init__(self, rr, start=0, end=5, points=51, 
        selections=None, steps=None):
        self.rr = rr
        self.start = start
        self.end = end
        self.points = points
        self.steps = steps
        self.species_ids = rr.getFloatingSpeciesIds()
        self.boundary_ids = rr.getBoundarySpeciesIds()
        self.reaction_ids = rr.getReactionIds()
        self.flux_ids = [x + "'" for x in self.species_ids]
        self.selections = selections
        self.set_selections()
        
        self.conditions = None
        self.simulations = None

    def set_selections(self):
        """
        Check selection values if provided by user. Otherwise, set selection 
        values as all floating species' amounts and derivatives and reaction rates.
        Time added to selections if it was omitted by user.
        """
        # Default behavior uses all floating species' amounts and derivatives
        # and reaction rates.
        if self.selections == None:
            self.selections = ['time'] + self.species_ids + self.flux_ids \
                + self.reaction_ids
        else:
            # Add time selection if omitted
            if 'time' not in self.selections:
                self.selections.insert(0, 'time')
                warnings.warn('Added time to list of selections.')
            # Check if selection is a 
            for x in self.selections:
                in_model = (x in self.species_ids) or (x in self.boundary_ids)
                if x != 'time' and not in_model:
                    raise ValueError(
                        f"{x} is not a floating or boundary species in the model.")
            # If selections list was empty, call default behavior
            if len(self.selections)<2:
                self.selections = None
                self.set_selections()

    
class OneWayExperiment(Experiment):
    """
    Performs one-way experiment where simulations were performed while varying 
    one parameter (factor) over a range of values (levels).

    Parameters
    ----------
    param: str
        The name of parameter to change in the roadRunner object.
    range: array_like
        The starting and ending values of sequence of levels to test for the
        parameter of interest.
    num: int (default: 10)
        Number of levels to test for the parameter of interest.
    levels: array_like (optional, default: None)
        Sequence of values to test for the parameter of interest. 
        If specified, overrides range and num arguments.
    {simulation_params}

    """
    def __init__(self, rr, param, range=(0,10), num=10, levels=None, **kwargs):
        # initialize attributes related to simulation
        super().__init__(rr, **kwargs)

        # Initialize attributes related to varying parameter
        self.param = param
        self.check_param()
        self.range = range
        self.num = num
        self.levels = levels
        self.set_conditions()
        return

    def check_param(self):
        """Check if parameter is specified in model."""
        if self.param not in self.rr.model.keys():
            raise ValueError(f"{self.param} not specified in model.")

    def set_conditions(self):
        """Generate conditions based on user input."""
        # If levels 
        if not self.levels:
            if len(self.range)!= 2:
                raise ValueError("Please define range by two values.")
            for i in self.range:
                if type(i) not in [int, float]:
                    raise ValueError("Start/End value must be numerical.")
            self.conditions = np.linspace(*self.range, num=self.num)
        else:
            try:
                self.conditions = np.array(self.levels, dtype=float)
            except:
                raise ValueError("Levels must be numerical.")
        return

    def iter_conditions(self, func, **kwargs):
        """Wrapper function for iterating through conditions."""
        outputs = []
        for value in self.conditions:
            # Reset model
            self.rr.reset()
            # Change parameter value
            self.rr[self.param] = value
            output = func(**kwargs)
            outputs.append(output)
        return outputs
    
    def _simulate(self):
        output = self.rr.simulate(
                        start=self.start, end=self.end, 
                        points=self.points, steps=self.steps, 
                        selections=self.selections)
        return output

    # def simulate(self):
    #     """Run and store simulations for conditions"""
    #     self.simulations = []
    #     for value in self.conditions:
    #         # Reset model
    #         self.rr.reset()
    #         # Change parameter value
    #         self.rr[self.param] = value
    #         output = self.rr.simulate(
    #             start=self.start, end=self.end, 
    #             points=self.points, steps=self.steps, 
    #             selections=self.selections)
    #         self.simulations.append(output)
    #     return

    def simulate(self):
        """Run and store simulations for each condition."""
        self.simulations = self.iter_conditions(self._simulate)
        return

    def _steady_state(self):
        self.rr.steadyStateSelections = self.steady_state_selections
        try:
            self.rr.steadyState()
        except:
            self.rr.conservedMoietyAnalysis = True 
            # Changing conservedMoietyAnalysis resets steadyStateSelections
            self.rr.steadyStateSelections = self.steady_state_selections
            self.rr.steadyState()
        output = self.rr.getSteadyStateValues()
        self.rr.conservedMoietyAnalysis = False # reset to default
        return output

    def calc_steady_state(self):
        """
        Run and store steady state values of floating species amount
        and reaction rates for each condition.
        """
        self.steady_state_selections = [x for x in self.selections \
            if (x not in self.flux_ids) and (x != 'time')]
        self.steady_states = np.vstack(self.iter_conditions(self._steady_state))
        return

    
            

            


    

