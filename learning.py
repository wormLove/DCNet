import torch
from collections import defaultdict
from abc import ABC, abstractmethod

class Organizer(ABC):
    """A template class to specify different types of learning rules as organizers
        Values: Learning hyperparameters
            lr: A float value specifying the learning rate
            lr_decay:  A float value specifying the decay in the learning rate
            beta: A float value specifying the memory in moving window averaging
            penalty: A float value specifying relating anti-hebbian penalty
    """
    def __init__(self, **kwargs):
        self.lr = kwargs.get('lr', 0.99)
        self.lr_decay = kwargs.get('lr_decay', 0.8)
        self.beta = kwargs.get('beta', 0.99)
        self.penalty = kwargs.get('penalty', 1.0)
        
    @abstractmethod
    def step(self):
        """Stores and updates potentials after each input
        """
        pass
    
    @abstractmethod
    def organize(self):
        """Transforms potentials into connection weights
        """
        pass
    
    @staticmethod
    @abstractmethod
    def _potential():
        """Calculates potential based on inputs and outputs
        """
        pass

class DiscriminationOrganizer(Organizer):
    """Class to implement local learning in discrimination layers
        Values:
            potential_hebb: 2D tensor to store the hebbian potentials
            potential_antihebb: 2D tensor to store the anti-hebbian potentials
            dropout: 1D boolean tensor indicating units to dropout
            counter: Total counts of the organize method
    """
    def __init__(self, out_dim: int, in_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.potential_hebb = torch.zeros(in_dim, out_dim)
        self.potential_antihebb = torch.zeros(out_dim, out_dim)
        self.dropout = torch.tensor([True]*out_dim)
        self.counter = 1
        
    def step(self, input: torch.Tensor, output: torch.Tensor):
        """Function to transform the layer activity and update potentials
            Args:
                output: 1D tensor of current layer activity
                input: Crresponding input tensor to the layer
            Return:
                Nothing 
        """
        output = self._filter(output)
        output = self._normalize(output)
        
        self.potential_hebb = self.beta*self.potential_hebb + (1-self.beta)*self._potential(input, output)
        self.potential_antihebb = self.beta*self.potential_antihebb + (1-self.beta)*self._potential(output, output)
        

    def organize(self, weights: torch.Tensor):
        """Function to produce updated weights based on potentials and previous weights
            Args:
                weights: previous weights
            Return:
                updated_weights: 2D tensor of updated weights
        """
        correction_factor = 1/(1 - self.beta**self.counter)
        updated_weights = (1 - self.lr)*weights + correction_factor*self.lr*(self.potential_hebb - torch.mm(weights, self.potential_antihebb))
        self.counter += 1
        self.lr = self.lr_decay*self.lr
        
        return updated_weights
    
    @staticmethod
    def _potential(x: torch.Tensor, y: torch.Tensor):
        return torch.mm(x.T, y)

    def _filter(self, output: torch.Tensor):
        """Function to droput previously active cells and update the indices that need to be dropped in the next iteration
            Args:
                output: 1D tensor of current layer activity
            Return:
                output: Current layer activity with dropout units removed
        """
        output = self.dropout.logical_not()*output
        self.dropout = output > 0
        return output

    @staticmethod
    def _normalize(output: torch.Tensor):
        """Function to normalize layer activity to unit norm
            Args:
                output: 1D tensor of current layer activity
            Return:
                Normalized activity
        """
        output_norm = torch.norm(output, p='fro').item()
        output = output/output_norm if output_norm != 0 else torch.zeros_like(output)
        return output


class ClassificationOrganizer(Organizer):
    """Class to implement local learning the classification layers
        Values:
            potential: The recurrent potential in the layer
    """
    def __init__(self, out_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.potential = torch.zeros(out_dim, out_dim)
        
    def step(self, output: torch.Tensor):
        """Function to transform the layer activity and update potentials
            Args:
                output: 1D tensor of current layer activity
            Return:
                Nothing 
        """
        output = self._transform(output)
        potential = self._potential(output)
        self.potential += (potential >= 0)*potential + self.penalty*(potential < 0)*potential
        
    def organize(self):
        """Function to produce updated weights based on potentials
            Return:
                updated_weights: 2D tensor of updated weights
        """
        updated_weights = (self.potential > 0).float()
        self.potential.fill_(0.0)
        return updated_weights
    
    def prune(self, weights: torch.Tensor):
        graph_dict = {idx: row.nonzero().flatten().tolist() for idx, row in enumerate(weights.fill_diagonal_(0.0))}
        return self._prune(torch.zeros_like(self.potential).fill_diagonal_(1.0), graph_dict)

    def _merging_step(self, graph_dict: dict):
        _merged_to = defaultdict(list)
        for k, v in graph_dict.items():
            if len(v) == 1:
                _merged_to[v[0]].append(k)
        graph_dict = self._remove_end_nodes(graph_dict)
        self._update_graph(graph_dict, _merged_to) if bool(graph_dict) else None
        return graph_dict, _merged_to
    
    def _pruning_step(self, t: torch.Tensor, pruning_dict: dict):
        for k, v in pruning_dict.items():
            for item in v:
                t[:,k] = (t[:,k]).logical_or(t[:,item])
                t[:,item].fill_(0.0)
    
    def _prune(self, pruned_tensor: torch.Tensor, graph_dict: dict):
        while bool(graph_dict):
            graph_dict, _merged_to = self._merging_step(graph_dict)
            self._pruning_step(pruned_tensor, _merged_to)
            if not bool(_merged_to):
                break
        self._pruning_step(pruned_tensor, graph_dict) if bool(graph_dict) else None
        return pruned_tensor
        
    @staticmethod
    def _potential(x: torch.Tensor):
        return torch.floor(torch.mm(x.T, x))
        
    @staticmethod
    def _transform(output: torch.Tensor):
        """Function to perform a non-linear transformation of the layer activity
            Args:
                output: 1D tensor of current layer activity
            Return:
                Transformed output
        """
        out_f_transformed = -0.5*torch.ones_like(output)
        out_f_transformed[output > 0] = 1.0
        return out_f_transformed
    
    @staticmethod
    def _remove_end_nodes(graph_dict: dict):
        return {k: v for k, v in graph_dict.items() if len(v) != 1}
    
    @staticmethod
    def _update_graph(graph_dict: dict, _merged_to: dict):
        for k, v in _merged_to.items():
            if k in graph_dict:
                for item in v:
                    graph_dict[k].remove(item)

