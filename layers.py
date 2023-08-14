import torch
from torch import nn
from torch.linalg import matrix_rank
from torch.nn.functional import normalize

from initialization import Initializer
from learning import DiscriminationOrganizer, ClassificationOrganizer

class LayerThresholding(nn.Module):
    """Custom activation function based on the layer response
    """
    def __init__(self, alpha: float=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, input: torch.Tensor):
        return input*(input > self.alpha*torch.std(input).item())


class Feedforward(nn.Module):
    """Linear layer initialized with unit norm weights
    """
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        assert weights.dim() == 2, "connectivity matrix must be 2D"
        self.weights = normalize(weights, p=2, dim=0)
    
    def forward(self, input: torch.Tensor):
        return torch.mm(input, self.weights)
    
    def update(self, weights: torch.Tensor, unit_norm: bool=True):
        """A function to unpdate the wights of the layer and optionally normalize columns to unit norm
            Args: 
                weights: A 2D tensor of updated weights
                unit_norm: A bool indication if the columns need to be normalized
            Return:
                Nothing
        """
        self.weights =  normalize(weights, p=2, dim=0) if unit_norm else weights


class Recurrent(nn.Module):
    """Recurrent layer that solves input = weight*output
    """
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        assert weights.dim() == 2 and weights.shape[0] == weights.shape[1], "recurrent connections must be all to all and 2D"
        self.weights = weights
    
    def forward(self, input: torch.Tensor):
        approx_recurrence = self._approximate_recurrence(input)
        return torch.mm(input, approx_recurrence)
    
    def update(self, weights: torch.Tensor):
        """A function to unpdate the wights of the layer
            Args: 
                weights: A 2D tensor of updated weights
            Return:
                Nothing
        """
        self.weights =  weights
    
    def _approximate_recurrence(self, input: torch.Tensor):
        """ Function to approximate the effect of recurrence or solve the linear equation input = weight*output
            Args:
                input: A 1D torch tensor corresponding to the input
            Return:
                system_matrix: A 2D tensor to solve the stated linear equation
        """
        # check shape constrains and initialize selection variables
        m = input.shape[1]
        selection_diag = torch.zeros(m)
        selection_indx = torch.flatten(torch.argsort(input, descending=True), start_dim=0)
        
        # perform binary search for the best selction diagonal
        left, right = 0, m
        while left < right:
            mid = int(0.5*(left+right))
            selection_diag[selection_indx[:mid+1]] = 1
            sys_mat = self._system_matrix(selection_diag)
            if matrix_rank(sys_mat) < m:
                right = mid
            else:
                left = mid+1
            selection_diag[torch.nonzero(selection_diag)] = 0
        selection_diag[selection_indx[:left]] = 1 
        
        # return the best selection diagonal
        return torch.inverse(self._system_matrix(selection_diag))
    
    def _system_matrix(self, selection_diag: torch.Tensor):
        """Function to calculate a system matrix that solves the linear equation
            Args: 
                selection_diag: A 1D vector having 1 at indices corresponding to active units
            Return:
                system_matrix: A 2D tensor
        """
        return torch.eye(len(selection_diag)) + torch.mm(torch.diag(selection_diag), self.weights) - torch.diag(selection_diag)


class DiscriminationModule(nn.Module):
    """Discrimination module comprising of a linear layer, a recurrent layer and a layer thresholding activation
    """
    def __init__(self, out_dim: int, initializer: Initializer, **kwargs):
        super().__init__()
        self.feedforward = Feedforward(initializer.weights(out_dim))
        self.recurrent = Recurrent(self.recurrent_weights)
        self.activation = LayerThresholding(alpha=kwargs.get('alpha', 1.0))
        self.organizer = DiscriminationOrganizer(out_dim, initializer.in_dim, **kwargs)
        
    def forward(self, input: torch.Tensor):
        assert input.dim() == 2 and input.shape[0] == 1, "input must be a row vector"
        out_ = self.feedforward(input)
        out_ = self.recurrent(out_)
        out_f = self.activation(out_)
        self.organizer.step(input, out_f)
        return out_f
    
    def organize(self):
        """Function to form connections between input and output layer units 
        """
        updated_weights = self.organizer.organize(self.connections)
        self.feedforward.update(updated_weights)
        self.recurrent.update(self.recurrent_weights)
    
    def labels(self, label_idx: int):
        """Function to predict the label for each unit's tuning property
        """
        return torch.argmax(self.connections[-label_idx:,:], dim=0)
    
    @property
    def recurrent_weights(self):
        """Function to calculate the recurrent weights based on the feedforward weights
        """ 
        return torch.mm(self.connections.T, self.connections)
    
    @property
    def connections(self):
        """Function to get the weights of the feedforward layer
        """
        return self.feedforward.weights


class ClassificationModule(nn.Module):
    """Classification module comprising of two linear layers and a layer thresholding activation
    """
    def __init__(self, out_dim: int, initializer: Initializer, **kwargs):
        super().__init__()
        self.feedforward1 = Feedforward(initializer.weights(out_dim))
        self.feedforward2 = Feedforward(torch.eye(out_dim))
        self.activation = LayerThresholding(alpha=kwargs.get('alpha', 1.0))
        self.organizer = ClassificationOrganizer(out_dim, **kwargs)
        self._pruning = False
    
    def forward(self, input: torch.Tensor):
        assert input.dim() == 2 and input.shape[0] == 1, "input must be a row vector"
        out_ = self.feedforward1(input)
        out_ = self.feedforward2(out_)
        out_f = self.activation(out_)
        self.organizer.step(out_f)
        return out_f
    
    def organize(self):
        """Function to form excitatory connections among the output neurons
        """
        updated_weights = self.organizer.organize()
        self.feedforward2.update(updated_weights, unit_norm=False)
        if self._pruning:
            pruned_weights = self.organizer.prune(self.connections, updated_weights)
            self.feedforward1.update(pruned_weights, unit_norm=False)
    
    def pruning(self, value: str):
        if value == 'on':
            self._pruning = True
    
    @property
    def connections(self):
        """Function to get the connections among the output neurons
        """
        return self.feedforward1.weights
    
    @property
    def recurrent_weights(self):
        return self.feedforward2.weights