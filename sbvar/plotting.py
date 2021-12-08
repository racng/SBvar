# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def plot_mesh(X, Y, Z, kind='contourf', projection='2d', 
    **kwargs):
    """
    Plot three-dimensional data as 3D surface, 2D filled contour, or 2D contours.
    Parameters
    ----------
    X: np.array
        2D array of X coordinates
    Y: np.array
        2D array of Y coordinates
    Z: np.array
        2D array of Z coordinates
    kind: str
        Method of plotting 3D data.
        - surface: Plots 3D surface (requires projection='3d')
        - contourf: 
    projection: str
        Plot using "2d" or "3d" projection.
    kwargs: 
        Keywords arguments for matplotib.pyplot.contour, matplot.pyplot.contourf, 
        or Axes3D.plot_surface
    """
    if projection=='3d':
        fig, ax = plt.subplots(subplot_kw={"projection": projection})
    else:
        fig, ax = plt.subplots()

    if kind=='surface':
        if projection=='2d':
            raise ValueError("Plotting surface requires 3d projection.")
        ax.plot_surface(X, Y, Z, **kwargs)
    
    if kind=='contourf':
        ax.contourf(X, Y, Z, **kwargs)
    
    if kind=='contour':
        ax.contour(X, Y, Z, **kwargs)
    return fig, ax

