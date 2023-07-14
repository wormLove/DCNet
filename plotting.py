import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pl
from tqdm import tqdm
from layers import DiscriminationModule, ClassificationModule


def auto_reshape(n: int):
    nrows, ncols = 1, n
    for delta in range(n):
        nrows = 0.5*(-delta + (delta**2 + 4*n)**(0.5))
        if nrows.is_integer():
            nrows = int(nrows)
            ncols = int(n/nrows)
            break
    return nrows, ncols


def transform_coordinates(ax, x, y, x_off, y_off):
    """Function to transform coordinates from axis reference frame to figure reference frame
        Args:
            ax: pyplot axis object
            x: x coordinate in the axis reference frame
            y: x coordinate in the axis reference frame
            x_off: x offset of the axis in figure reference frame
            y_off: y offset of the axis in figure reference frame
        Return:
            Transformed coordinates 
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ylen = ymax - ymin
    xlen = xmax - xmin

    x_transform = x_off + 0.095 + x*0.19/xlen
    y_transform = y_off + 0.095 + y*0.19/ylen
    return x_transform, y_transform

def plot_nodes(ax, x, y, graph, nClust):
    """Function to plot line between nodes of the network graph
        Args:
            ax: pyplot axis object on which nodes are plotted
            x: a 1D tensor of x coordinates of nodes in the axis reference frame
            y: a 1D tensor of y coordinates of nodes in the axis reference frame
            graph: adjecency matrix of nodes to determine the connected nodes
            cClust: integer specifying number of distinct clusters
        Return:
            Nothing
    """
    plotted = []
    colors = pl.cm.gist_ncar(np.linspace(0, 0.9, nClust))
    color_index = 0
    node_plotted = False
    for node1, row in enumerate(graph):
        for node2 in range(len(row)):
            if row[node2] == 1:
                if node1 in plotted and node2 in plotted:
                    pass
                else:
                    node_plotted = True
                    ax.plot([x[node1].item(), x[node2].item()], [y[node1].item(), y[node2].item()], color=colors[color_index]) 
                    plotted.append(node1)
                    plotted.append(node2)
        color_index = color_index+1 if node_plotted else color_index
        node_plotted = False


def plot_labels(fig, ax, x, y, x_off, y_off, indices, wf):
    """Function to plot learned receptive field of each neuron in form of a graph
        Args:
            fig: pyplot figure object
            ax: pyplot axis objecr
            x: a 1D tensor of x coordinate of the node in axis reference frame
            y: a 1D tensor of y coordinate of the node in axis reference frame
            x_off: x offset of the axis in the figure reference frame
            y_off: y offset of the axis in the figure reference frame
            indices: 1D tensor of neuron indices to be plotted
            wf: 2D tensor of the connection matrix
        Return:
            Nothing
    """
    for i, coord in enumerate(zip(x,y)):
        xt, yt = transform_coordinates(ax, coord[0], coord[1], x_off, y_off)
        imax = fig.add_axes([xt, yt, 0.01, 0.01])
        imax.pcolormesh(wf[:,indices[i]].reshape(28,28))
        imax.axis('off')


def coordinates(n: int):
    """Function to generate cicular coordinate values
        Args:
            n: number of points (integer)
        Return:
            Tuple of cicular x an y coordinate tensors
    """
    pi = torch.arccos(torch.tensor(-1)).item()
    theta = 2*pi*torch.arange(n)/n
    x = torch.cos(theta)
    y = torch.sin(theta)
    return x, y

def plot_clusters(wc: torch.Tensor, wd: torch.Tensor):
    """Function to plat custers of varying dgrees in form of a connected graph
        Args:
            module_c: A ClassificationModule object
            module_d: A DiscriminationModule object
        Return:
            Nothing
    """
    wc = wc.fill_diagonal_(0.0)
    all_degrees = torch.sum(wc, dim=0)
    unique_degrees = torch.unique(all_degrees).int().tolist()
    N = len(unique_degrees)

    fig = plt.figure(figsize=(25, 25))

    for k, degree in enumerate(unique_degrees):
        if degree > 0:
            degree_indices = (all_degrees == degree).nonzero().squeeze()
            nAtoms = len(degree_indices)
            nClust = int(nAtoms/(degree+1))
            graph = wc.index_select(0, degree_indices).index_select(1, degree_indices)
            

            x, y = coordinates(len(degree_indices))
            i, j = divmod(k,5)
            x_off = 0.01 + 0.2*j
            y_off = 0.01 + 0.2*i
            ax = fig.add_axes([x_off, y_off, 0.19, 0.19])  
            plot_nodes(ax, x, y, graph, nClust)
            plot_labels(fig, ax, x, y, x_off, y_off, degree_indices, wd)
            ax.set_title(f"{degree=}, {nAtoms=}")
            ax.axis('off')

def plot_connections(data: torch.Tensor):
    """Function to plot a matrix as image grid of its columns. Each column is reshaped into appropriate image dimensions
        Args:
            data: 2D data tensor
        Return:
            Nothing
    """
    assert data.dim() == 2, "data must be 2D tensor"
    panel_rows, panel_cols = auto_reshape(data.shape[1])
    atom_rows, atom_cols = auto_reshape(data.shape[0])
    fig, ax = plt.subplots(panel_rows, panel_cols, figsize=(panel_cols, panel_rows))
    for k in range(data.shape[1]):
        i,j = divmod(k, panel_cols)
        ax[i,j].pcolormesh(data[:,k].reshape(atom_rows, atom_cols))
        ax[i,j].axis('off')