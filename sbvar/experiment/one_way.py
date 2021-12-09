import numpy as np
import pandas as pd
import warnings

from sbvar.experiment import Experiment
from sbvar.utils import *
from sbvar.plotting import *

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