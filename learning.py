import torch
from abc import ABC, abstractmethod
from typing import Dict
from torch.nn.functional import normalize

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


class TripletOrganizer(Organizer):
    def __init__(self, out_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.margin = kwargs.get('margin', 1.0)
        self.potential = torch.zeros(out_dim, out_dim)
        self.embeddings = torch.empty(0)
        self.loss = 0.0
    
    
    def step(self, output: torch.Tensor):
        self._accumulate(output)
        try:
            emb_anc, emb_pos, emb_neg = self.embeddings[0].unsqueeze(0), self.embeddings[1].unsqueeze(0), self.embeddings[2].unsqueeze(0)
            if self._loss(emb_anc, emb_pos, emb_neg) > 0:
                self.potential = self.beta*self.potential + (1 - self.beta)*(self._potential(emb_anc, emb_pos) - self._potential(emb_anc, emb_neg))
            self.embeddings = torch.empty(0)
        except IndexError:
            pass
            
    def organize(self, weights: torch.Tensor):
        updated_weights = weights + self.lr*torch.mm(weights, self.potential)
        running_loss = self.loss
        self.loss = 0.0
        return updated_weights, running_loss 
    
    def reset(self):
        self.potential.fill_(0.0)
        self.embeddings = torch.empty(0)
        self.loss = 0.0
    
    def _accumulate(self, output: torch.Tensor):
        self.embeddings = torch.cat((self.embeddings, self._normalize(output)), dim=0)
        
    def _loss(self, emb_anc: torch.Tensor, emb_pos: torch.Tensor, emb_neg: torch.Tensor):
        d_pos = 1 - torch.mm(emb_anc, emb_pos.T).item()
        d_neg = 1 - torch.mm(emb_anc, emb_neg.T).item()
        loss = d_pos - d_neg + self.margin
        self.loss += loss
        return loss
    
    @staticmethod
    def _potential(x: torch.Tensor, y: torch.Tensor):
        return torch.mm(x.T, y)
    
    @staticmethod
    def _normalize(output: torch.Tensor):
        """Function to normalize layer activity to unit norm
            Args:
                output: 1D tensor of current layer activity
            Return:
                Normalized activity
        """
        return normalize(output, p=2, dim=1)
    
    
