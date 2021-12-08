import numpy as np

def meshes_to_meshvector(mesh_list):
    """
    Reshape 2D mesh into tidy meshvector. 
    Parameters
    ----------
    mesh_list: list of arrays
        List of n mesh grids with matching dimension (ixj).
    Returns
    -------
    mesh_vector: np.array 
        Reshaped 2D array with ixj rows and n columns. 
    """
    return np.array(mesh_list).reshape(len(mesh_list), -1).T

def vector_to_mesh(vector, dim1, dim2):
    """
    Reshape 1D vector to mesh form
    Parameters
    ----------
    vector: np.array
        1D vector to be reshaped
    Returns
    -------
    mesh: np.array
        Reshaped 2D mesh with `dim1` columns and `dim2` rows.
        This is consistent with np.meshgrid where the first dimension 
        varies on the column axis. 
    """
    return np.array(vector).reshape((dim2, dim1))

def meshvector_to_meshes(mesh_vector, dim1, dim2):
    """
    Reshape meshvector into list of meshes.
    Parameters
    ----------
    mesh_vector: np.array
        2D array with dim1 x dim2 rows and n columns.
    Returns
    -------
    mesh_list: list of np.array
        List of n mesh grids, each with `dim1` columns and `dim2` rows.
    """
    n = mesh_vector.shape[1]
    return [vector_to_mesh(mesh_vector[:, i], dim1, dim2) for i in range(n)]

