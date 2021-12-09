from matplotlib.pyplot import viridis
import numpy as np
from numpy.core.fromnumeric import var
import pandas as pd
from roadrunner import RoadRunner
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
        selections=None, steps=None, steady_state_selections=None):
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
        self.steady_state_selections = steady_state_selections
        self.set_steady_state_selections()
        
        self.dim = 0
        self.conditions = None
        self.simulations = None
        self.steady_states = None

    def check_in_model(self, x):
        in_model = (x in self.species_ids) or (x in self.boundary_ids) \
            or (x in self.reaction_ids) or (x in self.flux_ids)
        return in_model

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
                in_model = self.check_in_model(x)
                if x != 'time' and not in_model:
                    raise ValueError(f"{x} is not in the model.")
            # If selections list was empty, call default behavior
            if len(self.selections)<2:
                self.selections = None
                self.set_selections()

    def set_steady_state_selections(self):
        # Default behavior uses all floating species' amounts and reaction rates.
        if self.steady_state_selections == None:
            self.steady_state_selections = [x for x in self.selections \
                if (x not in self.flux_ids) and (x != 'time')]
        else:
            for x in self.steady_state_selections:
                if x != 'time' and not self.check_in_model(x):
                    raise ValueError(f"{x} is not in the model.")
        return 

    def iter_conditions(self, func, **kwargs):
        """
        Wrapper function for resetting model and calling `func`.
        Parent Experiment class does not have multiple conditions,
        so function is applied once to initial conditions.
        """
        # Reset model
        self.rr.reset()
        output = func(**kwargs)
        return output
    
    def _simulate(self):
        output = self.rr.simulate(
                        start=self.start, end=self.end, 
                        points=self.points, steps=self.steps, 
                        selections=self.selections)
        return output

    def simulate(self):
        """Run and store simulations for each condition."""
        self.simulations = np.dstack(self.iter_conditions(self._simulate))
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
        self.steady_states = np.vstack(self.iter_conditions(self._steady_state))
        return

    def get_selection_index(self, variable):
        """Get index of variable in selections."""
        if variable not in self.selections:
            raise ValueError(f"{variable} not in steady state selections.")
        return self.selections.index(variable)

    def get_steady_state(self, variable):
        """Get steady state values."""
        if variable not in self.steady_state_selections:
            raise ValueError(f"{variable} not in steady state selections.")
        if self.steady_states is None:
            warnings.warn("Calculating steady state.", UserWarning)
            self.calc_steady_state()
        i = self.steady_state_selections.index(variable)
        vector = self.steady_states[:, i]
        return vector

    def get_step_values(self, variable, step):
        """Get values of variable from time step."""
        i = self.get_selection_index(variable)
        if self.simulations is None:
            warnings.warn("Running simulations.")
            self.simulate()
        vector = self.simulations[step, i, :]
        return vector

    def get_timepoints(self):
        """Get array of all timepoints."""
        return self.simulations[:, 0, 0]

    def get_closest_timepoint(self, time):
        """Get index of timepoint closest to t."""
        return np.argmin(np.abs(self.get_timepoints()-time))

    def get_time_values(self, variable, time):
        """Get values of variable from timepoint closest to time."""
        i = self.get_selection_index(variable)
        if self.simulations is None:
            warnings.warn("Running simulations.")
            self.simulate()
        step = self.get_closest_timepoint(time)
        vector = self.get_step_values(variable, step)
        return vector

    def get_values(self, variable, steady_state=True, step=None, time=None):
        """Get meshgrid of simulation results for a variable. 
        If steady_state is True, returns steady state value. Otherwise,
        return variable value at a specific time or step. If time is provided,
        the nearest time point is returned.
        Parameters
        ----------
        steady_state: boolean
            If True, returns steady state values. Overrides step and time.
        step: int
            Index of time point to get values for.
        time: float
            Timepoint value to get values for. Values for the nearest
            timepoint is returned. 
        Return
        ------
        mesh: np.array
            2D Meshgrid of values. 
        """
        if steady_state:
            return self.get_steady_state(variable)
        elif step is not None:
            return self.get_step_values(variable, step)
        elif time is not None:
            return self.get_timepoint_values(variable, time)

class OneWayExperiment(Experiment):
    """
    Performs one-way experiment where simulations were performed while varying 
    one parameter (factor) over a range of values (levels).

    Parameters
    ----------
    param: str
        The name of parameter to change in the roadRunner object.
    bounds: array_like
        The starting and ending values of sequence of levels to test for the
        parameter of interest.
    num: int (default: 10)
        Number of levels to test for the parameter of interest.
    levels: array_like (optional, default: None)
        Sequence of values to test for the parameter of interest. 
        If specified, overrides bounds and num arguments.
    {simulation_params}

    """
    def __init__(self, rr, param, bounds=(0,10), num=10, levels=None, **kwargs):
        # initialize attributes related to simulation
        super().__init__(rr, **kwargs)

        # Initialize attributes related to varying parameter
        self.dim = 1
        self.param = param
        self.check_param()
        self.bounds = bounds 
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
        # If levels not provided, generate sequence of levels
        if not self.levels:
            if len(self.bounds)!= 2:
                raise ValueError("Please define bounds by two values.")
            for i in self.bounds:
                if type(i) not in [int, float]:
                    raise ValueError("Start/End value must be numerical.")
            self.conditions = np.linspace(*self.bounds, num=self.num)
        else:
            try:
                self.conditions = np.array(self.levels, dtype=float)
            except:
                raise ValueError("Levels must be numerical.")
        return

    def get_conditions_df(self):
        """Generate dataframe of conditions"""
        df = pd.DataFrame({self.param:self.conditions})
        return df
        
    def iter_conditions(self, func, **kwargs):
        """
        Wrapper function for iterating through each condition.
        For each condition, the model is reset and the parameter is updated
        before calling `func`. 
        Parameters
        ----------
        func: function
        kwargs: dict
            Dictionary of keyword arguments for `func`.
        Returns
        -------
        outputs: list
            list of outputs from applying `func` to each condition.
        """
        outputs = []
        for value in self.conditions:
            # Reset model
            self.rr.reset()
            # Change parameter value
            self.rr[self.param] = value
            output = func(**kwargs)
            outputs.append(output)
        self.rr.reset()
        return outputs

    def conditions_to_meshes(self):
        """
        Convert conditions into list of meshgrids.
        Returns list of meshgrid T and Y, where time is on the x-axis
        and param is on the y-axis.
        """
        t = self.get_timepoints()
        dim1 = len(t)
        dim2 =  len(self.conditions)
        T = np.tile(t, dim2).reshape((dim2, dim1))
        Y = np.tile(self.conditions, (dim1, 1)).T
        return T, Y

    def get_timecourse_mesh(self, variable):
        t = self.get_timepoints()
        dim1 = len(t)
        dim2 =  len(self.conditions)
        s = self.get_selection_index(variable)
        vector = np.concatenate([self.simulations[:, s, i] for i in range(dim2)])
        Z = vector_to_mesh(vector, dim1, dim2)
        return Z
    
    def plot_timecourse_mesh(self, variable, kind='contourf', projection='2d', 
        cmap='viridis', **kwargs):
        T, Y = self.conditions_to_meshes()
        Z = self.get_timecourse_mesh(variable)
        fig, ax, cax = plot_mesh(T, Y, Z, kind=kind, projection=projection, 
            cmap=cmap, **kwargs)
        ax.set_xlabel('Time')
        ax.set_ylabel(self.param)
        cax.set_title(variable)
        if projection=='3d':
            ax.set_zlabel(variable)
        else:
            ax.set_title(variable)
        return fig, ax

class TwoWayExperiment(Experiment):
    """
    Performs N-way experiment where simulations were performed while varying 
    two parameter (factor) over a range of values (levels).

    Parameters
    ----------
    param1: str
        The name of the first parameter to change in the roadRunner object.
    param2: str
        The name of the second parameter to change in the roadRunner object.
    bounds1: array_like
        The starting and ending values of sequence of levels to test for the
        first parameter.
    bounds2: array_like
        The starting and ending values of sequence of levels to test for the
        second parameter.
    num1: int (default: 10)
        Number of levels to test for the first parameter.
    num2: int (default: 10)
        Number of levels to test for the second parameter.
    levels1: array_like (optional, default: None)
        Sequence of values to test for the first parameter. 
        If specified, overrides bounds and num arguments.
    levels2: array_like (optional, default: None)
        Sequence of values to test for the second parameter. 
        If specified, overrides bounds and num arguments.
    {simulation_params}

    """
    def __init__(self, rr, param1, param2, 
        bounds1=(0,10), bounds2=(0,10), num1=10, num2=10, 
        levels1=None, levels2=None, **kwargs):
        # initialize attributes related to simulation
        super().__init__(rr, **kwargs)

        # Initialize attributes related to varying parameter
        self.dim = 2
        self.param_list = [param1, param2]
        self.check_params()
        self.bounds_list = [bounds1, bounds2]
        self.num_list = [num1, num2]
        self.levels_list = [levels1, levels2]
        self.set_conditions()
        return

    def check_params(self):
        """Check if parameter is specified in model."""
        for param in self.param_list:
            if param not in self.rr.model.keys():
                raise ValueError(f"{param} not specified in model.")

    def set_conditions(self):
        """
        Generate conditions based on user input.
        The determined levels for each parameter are stored in
        `conditions_list`. Conditions (combinations of parameter values) 
        are stored as list of meshgrids in `mesh_list` and as reshaped 
        meshvector in `conditions`. 
        """
        self.conditions_list = []
        for i in range(2):
            levels = self.levels_list[i]
            bounds = self.bounds_list[i]
            num = self.num_list[i]
            # If levels not provided, generate sequence of levels
            if not levels:
                if len(bounds)!= 2:
                    raise ValueError("Please define bounds by two values.")
                for i in bounds:
                    if type(i) not in [int, float]:
                        raise ValueError("Start/End value must be numerical.")
                conditions = np.linspace(*bounds, num=num)
            else:
                try:
                    conditions = np.array(self.levels, dtype=float)
                except:
                    raise ValueError("Levels must be numerical.")
            self.conditions_list.append(conditions)
        self.mesh_list = np.meshgrid(*self.conditions_list)
        # Convert meshgrid into long meshvector format
        self.conditions = meshes_to_meshvector(self.mesh_list)
    
    def get_conditions_df(self):
        """Generate dataframe of conditions"""
        df = pd.DataFrame(self.conditions, columns=self.param_list)
        return df

    def conditions_to_meshes(self):
        """
        Convert conditions into list of meshgrids.
        Returns list of meshgrid X and Y, where param1 is on the x-axis
        and param2 is on the y-axis.
        """
        dim1 = len(self.conditions_list[0])
        dim2 = len(self.conditions_list[1])
        return meshvector_to_meshes(self.conditions, dim1, dim2)

    def vector_to_mesh(self, v):
        """
        Convert vector into list of meshgrids.
        """
        dim1 = len(self.conditions_list[0])
        dim2 = len(self.conditions_list[1])
        return vector_to_mesh(v, dim1, dim2)

    def iter_conditions(self, func, **kwargs):
        """
        Wrapper function for iterating through each condition.
        For each condition, the model is reset and the parameter is updated
        before calling `func`. 
        Parameters
        ----------
        func: function
        kwargs: dict
            Dictionary of keyword arguments for `func`.
        Returns
        -------
        outputs: list
            list of outputs from applying `func` to each condition.
        """
        outputs = []
        for i, values in enumerate(self.conditions):
            # Reset model
            self.rr.reset()
            # Change parameter value
            for param, value in zip(self.param_list, values):
                self.rr[param] = value
            output = func(**kwargs)
            outputs.append(output)
        self.rr.reset()
        return outputs

    def get_steady_state(self, variable, mesh=True):
        """Get steady state value for variable for each condition."""
        vector = super().get_steady_state(variable)
        if mesh:
            return self.vector_to_mesh(vector)
        return vector
    
    def get_step_values(self, variable, step, mesh=True):
        """Get values of variable from time step."""
        vector = super().get_step_values(variable, step)
        if mesh:
            return self.vector_to_mesh(vector)
        return vector
    
    def get_time_values(self, variable, time, mesh=True):
        """Get values of variable from timepoint closest to time."""
        vector = super().get_time_values(variable, time)
        if mesh:
            return self.vector_to_mesh(vector)
        return vector
    
    def get_mesh(self, variable, steady_state=True, step=None, time=None):
        """Get meshgrid of simulation results for a variable. 
        If steady_state is True, returns steady state value. Otherwise,
        return variable value at a specific time or step. If time is provided,
        the nearest time point is returned.
        Parameters
        ----------
        steady_state: boolean
            If True, returns steady state values. Overrides step and time.
        step: int
            Index of time point to get values for.
        time: float
            Timepoint value to get values for. Values for the nearest
            timepoint is returned. 
        Return
        ------
        mesh: np.array
            2D Meshgrid of values. 
        """
        if steady_state:
            return self.get_steady_state(variable, mesh=True)
        elif step is not None:
            return self.get_step_values(variable, step, mesh=True)
        elif time is not None:
            return self.get_time_values(variable, time, mesh=True)
        
    
    def plot_mesh(self, variable, steady_state=True, step=None, time=None, 
        kind='contourf', projection='2d', cmap='viridis', **kwargs):
        """Plot simulation/calculation results as function of the two
        varying parameters.
        Parameters
        ----------
        steady_state: boolean
            If True, returns steady state values. Overrides step and time.
        step: int
            Index of time point to get values for.
        time: float
            Timepoint value to get values for. Values for the nearest
            timepoint is returned. 
        Return
        ------
        figure: matplotlib.figure
            Matplotlib figure object
        ax: matplotlib.axes 
            Matplotlib axes object
        """
        X, Y = self.conditions_to_meshes()
        Z = self.get_mesh(variable, steady_state=steady_state, step=step, 
            time=time)
        fig, ax, cax = plot_mesh(X, Y, Z, kind=kind, projection=projection, 
            cmap=cmap, **kwargs)
        ax.set_xlabel(self.param_list[0])
        ax.set_ylabel(self.param_list[1])
        cax.set_title(variable)
        if projection=='3d':
            ax.set_zlabel(variable)
        else:
            ax.set_title(variable)
        return fig, ax, cax

        


    
    


    
    

                


    

