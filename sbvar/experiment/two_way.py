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
        self.check_in_rr(param1)
        self.check_in_rr(param2)
        self.bounds_list = [bounds1, bounds2]
        self.num_list = [num1, num2]
        self.levels_list = [levels1, levels2]
        self.set_conditions()
        self.obs = self.get_conditions_df()
        return

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
            if  levels is None:
                if not isinstance(bounds, (list, tuple)):
                    raise TypeError("Bounds must be a sequence.")
                if len(bounds)!= 2:
                    raise ValueError("Please define bounds by two values.")
                for i in bounds:
                    if type(i) not in [int, float]:
                        raise ValueError("Start/End value must be numerical.")
                conditions = np.linspace(*bounds, num=num)
            else:
                if not isinstance(levels, (list, tuple)):
                    raise TypeError("Levels must be a sequence.")
                try:
                    conditions = np.array(levels, dtype=float)
                except:
                    raise ValueError("Levels must be numerical.")
            self.conditions_list.append(conditions)
        mesh_list = np.meshgrid(*self.conditions_list)
        # Convert meshgrid into long meshvector format
        self.conditions = meshes_to_meshvector(mesh_list)
    
    def get_conditions_df(self):
        """Generate dataframe of conditions
        
        Returns
        -------
        pd.DataFrame: Dataframe of conditions varying the two parameters, 
        with two columns denoting the levels of param1 and param2 respectively.
        """
        df = pd.DataFrame(self.conditions, columns=self.param_list)
        return df

    def conditions_to_meshes(self):
        """
        Convert conditions into list of meshgrids with Cartesian indexing.
        Returns list of meshgrid X and Y, where param1 varies along the x-axis
        and param2 on the y-axis.

        Returns
        -------
        tuple: Tuple of 2D mesh grid for `param1` and `param2`. 
            Each mesh has size n x m, where n is the number of levels for 
            `param2` and m is the number of levels for `param1`. 
        """
        dim1 = len(self.conditions_list[0])
        dim2 = len(self.conditions_list[1])
        return meshvector_to_meshes(self.conditions, dim1, dim2)

    def vector_to_mesh(self, v):
        """
        Convert a vector of values across conditions into meshgrid with 
        Cartesian indexing. Assumes vector is in the same order as `conditions`.

        Parameters
        ----------
        v: np.array
            1D np.array of values for each condition.

        Returns
        -------
        np.array: 2D mesh grid (n x m) of reshaped vector, 
            where n is the number of levels for `param2` and 
            m is the number of levels for `param1`. 
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

    def get_mesh(self, selection, steady_state=True, step=None, time=None):
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

        values = self.get_values(selection, steady_state=steady_state, 
            step=step, time=time)
        return self.vector_to_mesh(values)        
    
    def plot_mesh(self, selection, steady_state=True, step=None, time=None, 
        kind='contourf', projection='2d', cmap='viridis', **kwargs):
        """Plot simulation/calculation results as function of the two
        varying parameters.
        
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
        X, Y = self.conditions_to_meshes()
        Z = self.get_mesh(selection, steady_state=steady_state, step=step, 
            time=time)
        fig, ax, cax = plot_mesh(X, Y, Z, kind=kind, projection=projection, 
            cmap=cmap, **kwargs)
        ax.set_xlabel(self.param_list[0])
        ax.set_ylabel(self.param_list[1])
        cax.set_title(selection)
        if projection=='3d':
            ax.set_zlabel(selection)
        else:
            ax.set_title(selection)
        return fig, ax, cax