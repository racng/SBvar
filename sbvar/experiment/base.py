import numpy as np
import pandas as pd
import warnings

from sbvar.utils import *
from sbvar.plotting import *

doc_simulation = """
    rr : `RoadRunner` object
        RoadRunner object for simulating SBML model.
    start: float (default:0)
        Starting time for simulations
    end: float (default:5)
        Ending time for simulations
    points: int (default: 51)
        Number of time points to return for each simulation.
    steps: int (optional, default: None)
        Number of steps at which output is sampled, 
        where the samples are evenly spaced.
        Steps = points - 1. Steps and points may not both be specified. 
    selections: list of str (optional, default: None)
        A list of variables to be recorded in simulation output. 
        Time will be added to selections if not specified. If None, 
        keeps track of time and all floating species'amounts and derivatives 
        and reaction rates.
    conserved_moiety: bool
        If True, use conserved moiety analysis when calculating steady state.
"""

class Experiment(object):
    """
    Generic Experiment class for specifying model and simulation parameters.

    Parameters
    ----------
    {simulation_params}

    Attributes
    ----------
    conditions: list of tuples
        List of levels 
    simulations: numpy.ndarray
        Three-dimensional stacked arrays of simulation outputs across all conditions.
        (Number of timepoints x Number of Selections x Number of conditions)
    var: pd.DataFrame
        Dataframe storing annotation of selections
    obs: pd.DataFrame
        Dataframe storing annotations of conditions
    """
    def __init__(self, rr, start=0, end=5, points=51, 
        selections=None, steps=None, steady_state_selections=None, 
        conserved_moiety=False):
        self.rr = rr
        self.start = start
        self.end = end
        self.points = points
        self.steps = steps
        self.species_ids = rr.getFloatingSpeciesIds()
        self.boundary_ids = rr.getBoundarySpeciesIds()
        self.reaction_ids = rr.getReactionIds()
        self.flux_ids = [x + "'" for x in self.species_ids]
        self.set_selections(selections)
        # Note changing conservedMoietyAnalysis resets steadyStateSelections
        self.conserved_moiety = conserved_moiety
        self.set_steady_state_selections(steady_state_selections)
        
        self.dim = 0
        self.conditions = None
        self.simulations = None
        self.steady_states = None
        # TO-DO: populate with initial conditions
        self.obs = pd.DataFrame(self.conditions) 
        self.var = pd.DataFrame(self.selections)

    # def check_in_model(self, x):
    #     """"""
    #     in_model = (x in self.species_ids) or (x in self.boundary_ids) \
    #         or (x in self.reaction_ids) or (x in self.flux_ids)
    #     return in_model
    def check_in_rr(self, selection):
        """Check if selection is specified in the roadrunner object.
        Parameters
        ----------
        selection: str
            Selection variable to look for in the roadrunner object. 
        Return
        ------
        Returns nothing, but raises error if selection is not found.
        """
        # TO-DO: compare with self.rr.model.keys()?
        if selection not in self.rr.keys():
            raise ValueError(f"{selection} not in roadrunner object.")

    def set_selections(self, selections):
        """
        Set the selections attribute. Use user provided selections if 
        valide. Otherwise, set selection values to all floating species' 
        amounts and derivatives and reaction rates.
        Time added to selections if it was omitted by user.
        Parameters
        ----------
        selections: list of str
            List of variables to be recorded in simulation output.
        Returns
        -------
        Updates `selections` attribute. 
        """
        # Default behavior uses all floating species' amounts and derivatives
        # and reaction rates.
        if selections == None:
            self.selections = ['time'] + self.species_ids + self.flux_ids \
                + self.reaction_ids
        else:
            if type(selections)!=list:
                raise TypeError("Selections must be a list of strings.")
            self.selections = selections.copy()
            # Add time selection if omitted
            if 'time' not in self.selections:
                self.selections.insert(0, 'time')
                warnings.warn('Added time to list of selections.', UserWarning)
            # Check if selection is in model
            for x in self.selections:
                if x != 'time':
                    self.check_in_rr(x)
            # If selections list was empty, call default behavior
            if len(self.selections)<2:
                self.set_selections(None)

    def set_steady_state_selections(self, steady_state_selections):
        """
        Set the steady state selections attribute. 
        Use user provided selections if valid. Otherwise, set steady 
        state selection values to all floating species' amounts and 
        reaction rates.
        Parameters
        ----------
        steady_state_selections: list of str
            List of variables to be included in steady state calculations.
        
        Returns
        -------
        Updates `steady_state_selections` attribute. 
        """
        # Default behavior uses all floating species' amounts and reaction rates.
        if steady_state_selections == None:
            self.steady_state_selections = [x for x in self.selections \
                if (x not in self.flux_ids) and (x != 'time')]
        else:
            self.steady_state_selections = steady_state_selections.copy()
            for x in self.steady_state_selections:
                if x != 'time':
                    self.check_in_rr(x)
        return 

    def iter_conditions(self, func, **kwargs):
        """
        Wrapper function for resetting model and calling `func`.
        Parent Experiment class does not have multiple conditions,
        so function is applied once to initial conditions.

        Parameters
        ----------
        func: callable
            Function to be called for each condition.
        kwargs: dict
            Dictionary of keyword arguments for `func`.

        Returns
        -------
        output: list
            List containing output from calling `func` once.
        """
        # Reset model
        self.rr.reset()
        output = func(**kwargs)
        self.rr.reset()
        return [output]
    
    def _simulate(self):
        """Hidden function for running simulation for the current
        state of roadrunner `rr` attribute.

        Returns
        -------
        output: np.array
            Simulation output array with size (n x m), where n is the 
            number of timepoints and m is the number of selections.
        """
        output = self.rr.simulate(
                        start=self.start, end=self.end, 
                        points=self.points, steps=self.steps, 
                        selections=self.selections)
        return output

    def simulate(self):
        """Run simulations for each condition. Stored in `simulations`."""
        self.simulations = np.dstack(self.iter_conditions(self._simulate))
        return

    def _steady_state(self, approximate=False):
        """Hidden function for calculating steady state for the current
        state of roadrunner `rr` attribute. If calculation failed, return
        NaN array and suggest user to use conserved moiety analysis.

        Parameters
        ----------
        approximate: boolean
            If True, use approximation to find steady state. Useful if solver
            cannot find steady state.

        Returns
        -------
        output: np.array
            Steady state array with size (1 x m), where m is the 
            number of steady state selections.
        """
        self.rr.conservedMoietyAnalysis = self.conserved_moiety
        self.rr.steadyStateSelections = self.steady_state_selections
        if approximate:
            self.rr.steadyStateApproximate()
            output = self.rr.getSteadyStateValues()
        else:
            try:
                output = self.rr.getSteadyStateValues()
            except:
                if not self.conserved_moiety:
                    warnings.warn("Cannot calculate steady state." \
                        + "If model contains moiety conserved cycles, " \
                        + "set conserved_moiety to True.", UserWarning)
                output = np.tile(np.NaN, len(self.steady_state_selections))
        # Simulations fails if conserved moirty is True
        self.rr.conservedMoietyAnalysis = False # reset 
        return output

    def calc_steady_state(self, approximate=False):
        """
        Calculate steady state values of steady state selections for 
        each condition. Stored in `steady_states` attribute.

        Parameters
        ----------
        approximate: boolean (default: False)
            If True, use approximation to find steady state. Useful if solver
            cannot find steady state.
        """
        self.steady_states = np.vstack(self.iter_conditions(
            self._steady_state, approximate=approximate))
        return

    def get_selection_index(self, selection):
        """Get index of selection in `selections` attribute (a list).
        
        Parameters
        ----------
        selection: str
            Name of variable for which to get index.

        Returns
        -------
        int: Index of selection in the `selections` attribute.
        """
        if selection not in self.selections:
            raise ValueError(f"{selection} not in selections.")
        return self.selections.index(selection)

    def get_steady_state(self, selection):
        """Get steady state values across all conditions.
        
        Parameters
        ----------
        selection: str
            Name of variable for which to get steady state values.
        Returns
        -------
        vector: np.array
            1D array of steady state values of size n, where n is the 
            number of conditions.
        """
        if selection not in self.steady_state_selections:
            raise ValueError(f"{selection} not in steady state selections.")
        if self.steady_states is None:
            warnings.warn("Calculating steady state.", UserWarning)
            self.calc_steady_state()
        i = self.steady_state_selections.index(selection)
        vector = self.steady_states[:, i]
        return vector

    def get_step_values(self, selection, step):
        """Get time series values for a variable at a particular time step 
        across all conditions.
        
        Parameters
        ----------
        selection: str
            Name of variable for which to get time series values.
        step: int
            Index of time step in time series from which to get values.

        Returns
        -------
        vector: np.array
            1D array of time series values of size n, where n is the 
            number of conditions.
        """
        i = self.get_selection_index(selection)
        if self.simulations is None:
            warnings.warn("Running simulations.")
            self.simulate()
        vector = self.simulations[step, i, :]
        return vector

    def get_timepoints(self):
        """Get array of all timepoints. Requires running simulations to
        determine timepoints generated.
        Returns
        -------
        np.array: 1D array of time points of timeseries of size t, 
            where t is the number of `points` specified.
        """
        if self.simulations is None:
            warnings.warn("Running simulations.")
            self.simulate()
        return self.simulations[:, 0, 0]

    def get_closest_timepoint(self, time):
        """Get index of timepoint closest to specified time.
        
        Parameters
        ----------
        time: int, float
            Time value to match.

        Returns
        -------
            int: index of time point closest to the specified time. 
        """
        return np.argmin(np.abs(self.get_timepoints()-time))

    def get_time_values(self, variable, time):
        """Get time series values for a variable across all conditions
        at a particular time closest to specified time. 
        
        Parameters
        ----------
        selection: str
            Name of variable for which to get time series values.
        time: int
            Timepoint in time series from which to get the values.

        Returns
        -------
        vector: np.array
            1D array of time series values of size n, where n is the 
            number of conditions.
        """
        i = self.get_selection_index(variable)
        if self.simulations is None:
            warnings.warn("Running simulations.")
            self.simulate()
        step = self.get_closest_timepoint(time)
        vector = self.get_step_values(variable, step)
        return vector

    def get_values(self, variable, steady_state=True, step=None, time=None, 
        obs=False):
        """Get  array of simulation results for a variable. 
        If steady_state is True (default), returns steady state value. 
        Otherwise, return variable value at a specific time or step. 
        If time is provided, the nearest time point is returned. 
        Only one option can be specified.

        Parameters
        ----------
        steady_state: boolean
            If True, returns steady state values.
        step: int
            Index of time point to get values for.
        time: float
            Timepoint value to get values for. Values for the nearest
            timepoint is returned. 
        obs: True
            If True, look for column values in `obs` dataframe.
        Return
        ------
        vector: np.array
            1D array of values of size n, where n is the number of conditions.
        """
        if sum([steady_state, (step is not None), (time is not None), obs]) > 1:
            raise ValueError("Only one option can be specified.")
        if steady_state:
            return self.get_steady_state(variable)
        elif step is not None:
            return self.get_step_values(variable, step)
        elif time is not None:
            return self.get_time_values(variable, time)
        elif obs:
            return self.obs[variable].values


    def iter_timeseries(self, selection, func, **kwargs):
        """Iterate through each conditions and get timeseries of selection.
        
        Parameters
        ----------
        selection: str
            Name of variable for which to get timeseries values.
        func: callable
            Function applied to timeseries array.
        kwargs: dict
            Additional keyword arguments to for `func`.
        
        Returns
        -------
        list: list of output from applying function to timeseries of each
            condition
        """
        s = self.get_selection_index(selection)
        output = []
        for i, row in self.conditions.iterrows():
            ts = self.simulations[:, s, i]
            output.append(func(ts, **kwargs))
        return output

    