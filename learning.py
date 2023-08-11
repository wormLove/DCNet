import torch
from abc import ABC, abstractmethod

class Organizer(ABC):
    """A template class to specify different types of learning rules as organizers
        Values: Learning hyperparameters
            lr: A float value specifying the learning rate
            lr_decay:  A float value specifying the decay in the learning rate
            beta: A float value specifying the memory in moving window averaging
            penalty: A float value specifying relating anti-hebbian penalty
    """
    def __init__(self, lr: float=0.99, lr_decay: float=0.8,  beta: float=0.99, penalty: float=1.0):
        self.lr = lr
        self.lr_decay = lr_decay
        self.beta = beta
        self.penalty = penalty
        
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
    def __init__(self, out_dim: int, in_dim: int, lr: float=0.99, lr_decay: float=0.8,  beta: float=0.99):
        super().__init__(lr=lr, lr_decay=lr_decay, beta=beta)
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
    def __init__(self, out_dim: int, penalty: float=1.0):
        super().__init__(penalty=penalty)
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