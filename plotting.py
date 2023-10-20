import torch
from abc import ABC, abstractmethod
from IPython.display import clear_output, display
from typing import Dict, List, Callable, Union, Optional
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.nn.functional import normalize

class Plotter(ABC):
    def __init__(self, nplots: int, layout: Optional[Union[tuple, str]] = None):
        self.nplots = nplots
        self.layout = self._resolve_layout(layout)
        self.fig = plt.figure(figsize=(6*self.layout[1],5*self.layout[0]))
    
    def _resolve_layout(self, layout: Optional[Union[tuple, str]] = None):
        if layout is None:
            return 1, self.nplots
        elif layout == 'auto':
            return self.autoreshape(self.nplots)
        else:
            return layout
        
    def update_figure(self):
        clear_output(wait=True)
        display(self.fig)
        
    @staticmethod
    def autoreshape(n: int):
        nrows, ncols = 1, n
        for delta in range(n):
            nrows = 0.5*(-delta + (delta**2 + 4*n)**(0.5))
            if nrows.is_integer():
                nrows = int(nrows)
                ncols = int(n/nrows)
                break
        return nrows, ncols
        
    @abstractmethod
    def __call__(self):
        pass
    
    @abstractmethod
    def _initialize_figure(self):
        pass


class ProjectionPlotter(Plotter):
    def __init__(self, nplots: int = 1, layout: Optional[Union[tuple, str]] = None, cmap: str = 'tab10', reducer: Callable = PCA(n_components=2).fit_transform):
        super().__init__(nplots, layout)
        self.reducer = reducer
        self.cmap = cmap
        self._initialize_figure()
        
    def __call__(self, activations: Dict, targets: List[int], updates: bool = True):
        assert len(activations) == self.nplots
        assert len(targets) == activations[0].shape[0]
        
        for axis_idx in range(1, self.nplots+1):
            projections = self.reducer(activations[axis_idx-1])
            self._scatter(axis_idx, projections, targets)
        super().update_figure() if updates else None
    
    def _initialize_figure(self):
        for axis_idx in range(1, self.nplots+1):
            vars(self)['ax'+str(axis_idx)] = self.fig.add_subplot(*self.layout, axis_idx)
        plt.close(self.fig)
    
    def _scatter(self, axis_idx: int, projections: torch.Tensor, targets: List[int]):
        axis = vars(self)['ax'+str(axis_idx)]
        axis.clear()
        axis.scatter(projections[:,0], projections[:,1], c=targets, cmap=self.cmap)
        

class LossPlotter(Plotter):
    def __init__(self, nplots: int = 1, layout: Optional[Union[tuple, str]] = None):
        super().__init__(nplots, layout)
        self.loss = []
        self._initialize_figure()
        
    def __call__(self, loss: Union[float, List[float]], updates: bool = True):
        self.loss.append(loss)
        self.ax.clear()
        self.ax.plot(self.loss)
        super().update_figure() if updates else None
        
    def _initialize_figure(self):
        self.ax = self.fig.add_subplot(*self.layout, 1)
        plt.close(self.fig)
        

class MatrixPlotter(Plotter):
    def __init__(self, nplots: int = 1, layout: Optional[Union[tuple, str]] = None, cmap: str = 'viridis'):
        super().__init__(nplots, layout)
        self.cmap = cmap
        self._initialize_figure()
        
    def __call__(self, data: torch.Tensor, rows: List[int] = None, updates: bool = True):
        assert data.dim() == 2, 'input data must be a 2d tensor'
        assert self.nplots <= data.shape[0], 'plotted rows must be a subset of all rows'
        
        rows = [*range(data.shape[0])] if rows is None else rows
        for axis_idx in range(1, self.nplots+1):
            row = rows[axis_idx-1]
            self._mesh(axis_idx, data[row,:])
        super().update_figure() if updates else None
    
    def _mesh(self, axis_idx: int, data: torch.Tensor):
        axis =  vars(self)['ax'+str(axis_idx)]
        shape = super().autoreshape(len(data))
        axis.clear()
        axis.pcolormesh(data.reshape(shape), cmap=self.cmap)
        
    def _initialize_figure(self):
        for axis_idx in range(1, self.nplots+1):
            vars(self)['ax'+str(axis_idx)] = self.fig.add_subplot(*self.layout, axis_idx)
            vars(self)['ax'+str(axis_idx)].set_xticks([])
            vars(self)['ax'+str(axis_idx)].set_yticks([])
        plt.close(self.fig)
        
class SimilarityPlotter(Plotter):
    def __init__(self, nplots: int = 1, layout: Optional[Union[tuple, str]] = None, cmap: str = 'viridis'):
        super().__init__(nplots, layout)
        self.cmap = cmap
        self._initialize_figure()
    
    def __call__(self, data1: Dict, data2: Optional[Dict] = None, updates: bool = True):
        assert len(data1) == self.nplots, 'number of matrices must equal number of plots'
        
        for axis_id in range(1, self.nplots+1):
            similarity = self._similarity(data1[axis_id-1], data1[axis_id-1]) if data2 is None else self._similarity(data1[axis_id-1], data2[axis_id-1])
            self._mesh(axis_id, similarity)
        super().update_figure() if updates else None
        
    def _mesh(self, axis_idx: int, similarity: torch.Tensor):
        axis =  vars(self)['ax'+str(axis_idx)]
        axis.clear()
        axis.pcolormesh(similarity, cmap=self.cmap)
        
    @staticmethod
    def _similarity(x: torch.Tensor, y: torch.Tensor):
        x_normalized = normalize(x, p=2, dim=1)
        y_normalized = normalize(y, p=2, dim=1)
        return torch.mm(x_normalized, y_normalized.T)   
    
    def _initialize_figure(self):
        for axis_idx in range(1, self.nplots+1):
            vars(self)['ax'+str(axis_idx)] = self.fig.add_subplot(*self.layout, axis_idx)
        plt.close(self.fig)