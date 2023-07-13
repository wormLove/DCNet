import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pl
from tqdm import tqdm
from layers import DiscriminationModule, ClassificationModule

def run_model(n_samples, lr, alpha, init_connections, test_set):
    module_d = DiscriminationModule(init_connections, beta=0.98, lr=lr, alpha=alpha)
    module_c = ClassificationModule(torch.eye(1000))

    for i in tqdm(range(n_samples)):
        sample = test_set[:, i].T
        x = torch.unsqueeze(torch.tensor(sample).float(), 0)
        y_ = module_d(x)
        module_c(y_)

        module_d.organize() if i > 0 and i%50 == 0 else None 
        module_c.organize() if i > 0 and i%200 == 0 else None

    return module_d, module_c

def transform_coordinates(ax, x, y, x_off, y_off):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ylen = ymax - ymin
    xlen = xmax - xmin

    x_transform = x_off + 0.095 + x*0.19/xlen
    y_transform = y_off + 0.095 + y*0.19/ylen
    return x_transform, y_transform

def plot_nodes(ax, x, y, graph, nClust):
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
    for i, coord in enumerate(zip(x,y)):
        xt, yt = transform_coordinates(ax, coord[0], coord[1], x_off, y_off)
        imax = fig.add_axes([xt, yt, 0.01, 0.01])
        imax.pcolormesh(wf[:,indices[i]].reshape(28,28))
        imax.axis('off')


def coordinates(n: int, k: int, N: int):
    pi = torch.arccos(torch.tensor(-1)).item()
    theta = 2*pi*torch.arange(n)/n
    x = torch.cos(theta)
    y = torch.sin(theta)
    return x, y