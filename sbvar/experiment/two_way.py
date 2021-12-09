import numpy as np
import pandas as pd
import warnings

from sbvar.experiment import Experiment
from sbvar.utils import *
from sbvar.plotting import *

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