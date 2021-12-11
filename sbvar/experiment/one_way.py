import numpy as np
import pandas as pd
import warnings
from collections.abc import Iterable

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
    bounds: list or tuple
        The starting and ending values of sequence of levels to test for the
        parameter of interest.
    num: int (default: 10)
        Number of levels to test for the parameter of interest.
    levels: list or tuple (optional, default: None)
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
        self.check_in_rr(param)
        self.bounds = bounds 
        self.num = num
        self.levels = levels
        self.set_conditions()
        self.obs = self.get_conditions_df()
        return

    def set_conditions(self):
        """Generate conditions based on user input."""
        # If levels not provided, generate sequence of levels
        if self.levels is None:
            if not isinstance(self.bounds, (list, tuple)):
                raise TypeError("Bounds must be a sequence.")
            if len(self.bounds)!= 2:
                raise ValueError("Please define bounds by two values.")
            for i in self.bounds:
                if type(i) not in [int, float]:
                    raise TypeError("Start/End value must be numerical.")
            self.conditions = np.linspace(*self.bounds, num=self.num)
        else:
            if not isinstance(self.levels, (list, tuple)):
                raise TypeError("Levels must be a sequence.")
            try:
                self.conditions = np.array(self.levels, dtype=float)
            except:
                raise ValueError("Levels must be numerical.")
        return

    def get_conditions_df(self):
        """Generate dataframe of conditions
        
        Returns
        -------
        pd.DataFrame: Dataframe with the varying parameter conditions as 
            a column.
        """
        df = pd.DataFrame({self.param:self.conditions})
        return df
        
    def iter_conditions(self, func, **kwargs):
        """
        Wrapper function for iterating through each condition.
        For each condition, the model is reset and the parameter is updated
        before calling `func`. 

        Parameters
        ----------
        func: callable
            Function to be called for each condition.
        kwargs: dict
            Dictionary of keyword arguments for `func`.

        Returns
        -------
        outputs: list
            List of outputs from applying `func` to each condition.
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
        Convert conditions into list of meshgrids with Cartesian indexing.
        Returns list of meshgrid T and Y, where time is on the x-axis
        and param is on the y-axis.

        Returns
        -------
        tuple: Tuple of 2D mesh for time and the varying parameter. 
            Each mesh has size n x m, where n is the number of conditions
            and m is the number of time points. 
        """
        t = self.get_timepoints()
        dim1 = len(t)
        dim2 =  len(self.conditions)
        T = np.tile(t, dim2).reshape((dim2, dim1))
        Y = np.tile(self.conditions, (dim1, 1)).T
        return T, Y

    def get_timecourse_mesh(self, selection):
        """
        Get time courses of a variable across conditions as a meshgrid 
        with Cartesian indexing.

        Parameters
        ----------
        selection: str
            Name of variable for which to get time courses.

        Returns
        -------
        Z: np.array 
            2D meshgrid (n x m) of time courses where n is the
            number of conditions and m is the number of time pointss.
        """
        t = self.get_timepoints()
        dim1 = len(t)
        dim2 =  len(self.conditions)
        s = self.get_selection_index(selection)
        vector = np.concatenate([self.simulations[:, s, i] for i in range(dim2)])
        Z = vector_to_mesh(vector, dim1, dim2)
        return Z
    
    def plot_timecourse_mesh(self, selection, kind='contourf', projection='2d', 
        cmap='viridis', **kwargs):
        """
        Plot 3D time courses across all conditions. 
        
        Parameters
        ----------
        selection: str
            Name of variable to plot as time courses.
        kind: str {surface, contour, contourf}
            Method of plotting 3D timecourse.
        projection: str {'2d', '3d'}
            Projection of axes as 2D or 3D graph. surface requires 
        cmap: str (optional, default: 'viridis')
            Colormap for coloring selection values. 
        kwargs: dict
            Additional keyword arguments for matplotlib plot functions.

        Returns
        -------
        figure: matplotlib.figure
            Matplotlib figure object
        ax: matplotlib.axes 
            Matplotlib axes object
        cax: matplotlib.axes 
            Matplotlib colorbar axes object
        """
        T, Y = self.conditions_to_meshes()
        Z = self.get_timecourse_mesh(selection)
        fig, ax, cax = plot_mesh(T, Y, Z, kind=kind, projection=projection, 
            cmap=cmap, **kwargs)
        ax.set_xlabel('Time')
        ax.set_ylabel(self.param)
        cax.set_title(selection)
        if projection=='3d':
            ax.set_zlabel(selection)
        else:
            ax.set_title(selection)
        return fig, ax, cax

    def plot_line(self, selection, steady_state=True, step=None, time=None, 
        obs=False, **kwargs):
        """
        Plot trend of selection value as a function of the `param`. 
        
        Parameters
        ----------
        selection: str
            Name of variable to plot as time courses.
        steady_state: boolean
            If True, returns steady state values. Overrides step and time.
        step: int
            Index of time point to get values for.
        time: float
            Timepoint value to get values for. Values for the nearest
            timepoint is returned. 
        obs: boolean
            If True, get values from `obs` dataframe.
        kwargs: dict
            Additional keyword arguments for matplotlib plot functions.

        Returns
        -------
        figure: matplotlib.figure
            Matplotlib figure object
        ax: matplotlib.axes 
            Matplotlib axes object
        """
        fig, ax = plt.subplots()
        x = self.conditions
        y = self.get_values(selection, steady_state=steady_state, step=step, 
            time=time, obs=obs)
        plt.plot(x, y)
        plt.xlabel(self.param)
        plt.ylabel(selection)
        return fig, ax