from sklearn.cluster import KMeans
import matplotlib as mpl
import numpy as np

def cluster_timeseries(experiment, selection, method, **kwargs):
    """"Cluster timeseries of selection across conditions. 
    Cluster labels are added to experiment.obs dataframe with column name
    '{method}_{selection}'.

    Parameters
    ----------
    experiment: object
        OneWayExperiment or TwoWayExperiment class object.
    selection: str
        Name of variable for which to get timeseries values.
    method: str
        Clustering method. See each method for arugment requirements.

    kwargs: dict
        Additional keyword arguments for clustering function.
    Returns
    -------
    list: Cluster labels for each condition.
    """
    s = experiment.get_selection_index(selection)
    X = experiment.simulations[:, s, :].T
    if method == 'kmeans':
        labels = kmeans(X, **kwargs)

    experiment.obs[f"{method}_{selection}"] = labels
    return labels

def kmeans(X, n_clusters=4, init='k-means++', n_init=10, **kwargs):
    """
    Cluster samples in matrix (samples x features) using KMeans.
    
    Parameters 
    ----------
    X: np.array
        2D array of values with samples as rows and features as columns
    n_clusters: int (default: 10)
        The number of clusters to form.
    init: str (default: 'k-means++')
        Initialization method.
    n_init: int
        Number of times to run clustering algorithm
    kwargs: dict
        Additional keyword arguments for sklearn.cluster.Kmeans

    Returns:
        list: cluster index for each sample.
    """
    return KMeans(n_clusters=n_clusters, init=init, n_init=n_init, 
        **kwargs).fit_predict(X)
    
def plot_timecourse_clusters(experiment, selection, method, dashline=True, 
    **kwargs):
    """
    Plot contour plot of timeseries with additional colorbar for cluster labels. 

    Parameters
    ----------
    experiment: object
        OneWayExperiment or TwoWayExperiment class object.
    selection: str
        Name of variable for which to get timeseries values.
    method: str
        Clustering method. See each method for arugment requirements.
    dashline: boolean
        If True, plot dashed lines on the contour plot indicating cluster
        boundaries.
    Returns
    -------
    figure: matplotlib.figure
        Matplotlib figure object
    ax: matplotlib.axes 
        Matplotlib axes object
    """
    fig, ax, _ = experiment.plot_timecourse_mesh(selection, kind="contourf", 
        projection="2d", **kwargs)
    key = f"{method}_{selection}"
    v = experiment.obs[key].values
    cmap = mpl.cm.jet
    bounds = np.where(v[:-1] != v[1:])[0]
    bounds = np.insert(bounds, 0, 0)
    bounds = np.append(bounds, len(v)-1)
    bounds = [experiment.conditions[i] for i in bounds]
    if dashline:
        for b in bounds:
            ax.axhline(b, linestyle='--', color='k')
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    cax_new = ax.inset_axes([1.3,0,0.1, 1])
    cax_new.set_title(key)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ticks=bounds, spacing='proportional',
                cax=cax_new, orientation='vertical')
    return fig, ax
        


