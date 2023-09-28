import torch
from abc import ABC, abstractmethod
from typing import Dict

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
    
    @abstractmethod
    def reset(self):
        """Resets potential values back to zero
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
    
    def reset(self):
        self.potential_hebb.fill_(0.0)
        self.potential_antihebb.fill_(0.0)

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
        
    def organize(self, weights: torch.Tensor):
        """Function to produce updated weights based on potentials
            Return:
                updated_weights: 2D tensor of updated weights
        """
        updated_weights = weights.logical_or(self.potential > 0).float()
        #self.potential.fill_(0.0)
        self.reset()
        return updated_weights

    
    @staticmethod
    def _potential(x: torch.Tensor):
        return torch.floor(torch.mm(x.T, x))
    
    def reset(self):
        self.potential.fill_(0.0)
        
    @staticmethod
    def _transform(x: torch.Tensor):
        """Function to perform a non-linear transformation of the layer activity
            Args:
                output: 1D tensor of current layer activity
            Return:
                Transformed output
        """
        x_transformed = -0.5*torch.ones_like(x)
        x_transformed[x>0] = 1.0
        return x_transformed


class Pruner(Organizer):
    def __init__(self, out_dim: int, pruner_alpha: float):
        self.potential = torch.zeros(out_dim, out_dim)
        self.alpha = pruner_alpha
    
    def step(self, input: torch.Tensor, output: torch.Tensor):
        input_transformed = self._transform(input)
        output_transformed = self._transform(output)
        
        input_potential = self._potential(input_transformed)
        output_potential = self._potential(output_transformed)
        
        self.potential += input_potential.logical_xor(output_potential)
    
    def organize(self, weights: torch.Tensor):
        thereshold = self.alpha*torch.std(self.potential).item()
        weights_to_keep = self.potential.fill_diagonal_(0.0) <= thereshold
        #print(weights_to_keep.count_nonzero().item())
        updated_weights = weights.logical_and(weights_to_keep).float()
        #self.potential.fill_(0.0)
        self.reset()
        return updated_weights
    
    @staticmethod            
    def _potential(x: torch.Tensor):
        return torch.mm(x.T, x)
    
    def reset(self):
        self.potential.fill_(0.0)
    
    @staticmethod
    def _transform(x: torch.Tensor):
        return (x > 0).float()
    
class Teacher(Organizer):
    def __init__(self, out_dim: int, t_alpha: int):
        self.potential = torch.zeros(out_dim, out_dim)
        self.t_alpha = t_alpha
            
    def step(self, output: torch.Tensor):
        output = self._transform(output)
        self.potential = self._potential(output)
        
    def organize(self, weights: torch.Tensor):
        return weights.logical_or(self.potential).float()
        
    @staticmethod
    def _potential(x: torch.Tensor):
        return torch.mm(x.T, x)
    
    def reset(self):
        self.potential.fill_(0.0)
        
    #@staticmethod
    def _transform(self, x: torch.Tensor):
        x_transformed = torch.zeros_like(x)
        idxs = torch.argsort(x, descending=True).flatten()
        x_transformed[0,idxs[:self.t_alpha]] = 1.0
        return x_transformed
