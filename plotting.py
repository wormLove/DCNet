import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pl
from tqdm import tqdm


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

def connect_nodes(ax, x: torch.Tensor, y: torch.Tensor, color):
    m = (y[1] - y[0])/(x[1] - x[0])
    c = (y[0]*x[1] - y[1]*x[0])/(x[1] - x[0])

    xnew = np.linspace(x.min().item(), x.max().item(), num=50, endpoint=True)
    ynew = m*xnew + c
    
    cnorm = (xnew[25]**2 + ynew[25]**2)**(0.5)
    shiftx = xnew[25]/cnorm
    shifty = ynew[25]/cnorm
    for i in range(50):
        alpha = i*(50-i)*0.1*cnorm.item()/625
        xnew[i] -= alpha*shiftx
        ynew[i] -= alpha*shifty  
    
    ax.plot(xnew, ynew, color=color)
    

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

def plot_nodes(ax, graph, label_indices, connected_indices, graph_coord_dict):
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
    colors = pl.cm.gist_ncar(np.linspace(0, 0.9, len(label_indices))) # type: ignore
    cidx = 0
    for idx1 in label_indices:
        c = colors[cidx]
        for idx2 in connected_indices:
            if graph[idx1, idx2] == 1:
                if idx1 in plotted and idx2 in plotted:
                    pass
                else:
                    coord_x = torch.tensor([graph_coord_dict[idx1.item()][0], graph_coord_dict[idx2.item()][0]])
                    coord_y = torch.tensor([graph_coord_dict[idx1.item()][1], graph_coord_dict[idx2.item()][1]])
                    connect_nodes(ax, coord_x, coord_y, c)
                    plotted.append(idx1)
                    plotted.append(idx2)
        cidx += 1


def get_connected_indices(label_indices: torch.Tensor, graph: torch.Tensor):
    return (torch.sum(graph[label_indices,:], dim=0) > 0).nonzero().squeeze()

def generate_clusters(label_indices: torch.Tensor, graph: torch.Tensor, max_size: int=100):
    total_nodes = 0
    label_indices_trunc = []
    for i, idx in enumerate(label_indices):
        total_nodes += torch.sum(graph[idx, :])
        label_indices_trunc.append(idx.item())
        if total_nodes > max_size:
            total_nodes = torch.sum(graph[idx, :])
            yield torch.tensor(label_indices_trunc[:-1]), get_connected_indices(torch.tensor(label_indices_trunc[:-1]), graph)
            label_indices_trunc = []
            label_indices_trunc.append(idx.item())
    yield torch.tensor(label_indices_trunc), get_connected_indices(torch.tensor(label_indices_trunc), graph)

def plot_labels(fig, ax, x_off, y_off, graph_coord_dict, wf):
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
    nrows, ncols = auto_reshape(wf.shape[0])
    for idx, coord in graph_coord_dict.items():
        xt, yt = transform_coordinates(ax, coord[0], coord[1], x_off, y_off)
        imax = fig.add_axes([xt, yt, 0.01, 0.01])
        imax.pcolormesh(wf[:,idx].reshape(nrows, ncols))
        imax.axis('off')

def plot_clusters(wc: torch.Tensor, wd: torch.Tensor, atom_labels: torch.Tensor):
    """Function to plat custers of varying dgrees in form of a connected graph
        Args:
            module_c: A ClassificationModule object
            module_d: A DiscriminationModule object
        Return:
            Nothing
    """
    wc = wc.fill_diagonal_(0.0)
    unique_labels = torch.unique(atom_labels).tolist()
    for label in unique_labels:
        fig = plt.figure(figsize=(25, 25))
        cluster_cout = 0
        label_indices = (atom_labels == label).nonzero().squeeze()
        clusters = generate_clusters(label_indices, wc, max_size=50)
        for label_cluster, connected_cluster in clusters:
            try:
                all_indices = torch.cat([label_cluster, connected_cluster]).unique()
                nAtoms = len(all_indices)
            
                x, y = coordinates(len(all_indices))
                graph_coord_dict = {idx.item() : (x[i].item(), y[i].item()) for i, idx in enumerate(all_indices)}

                i, j = divmod(cluster_cout, 5)
                x_off = 0.01 + 0.2*j
                y_off = 0.01 + 0.2*i
                ax = fig.add_axes([x_off, y_off, 0.19, 0.19])

                plot_nodes(ax, wc, label_cluster, connected_cluster, graph_coord_dict) 
                plot_labels(fig, ax, x_off, y_off, graph_coord_dict, wd)
                ax.set_title(f"{label=},{nAtoms=}")
                ax.axis('off')
                cluster_cout += 1
            except StopIteration:
                pass


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