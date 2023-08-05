import torch

class Organizer:
    """Class to implement local learning in layers
        Values:
            out_dim: Dimension of the layer
            in_dim: Dimension of input to the layer
            lr: Learning rate
            lr_decay: Decay factor of the learning rate
            beta1: A constant to decide window size of moving average
            beta2: A constant to decide window size of moving average
            feedforward_potential: 2D tensor to store the potential for feedforward connections
            recurrent_potential: 2D tensor to store the potential for recurrent connections
            dropout: 1D boolean tensor indicating units to dropout
            counter: Total counts of the organize method
    """
    def __init__(self, out_dim: int, in_dim: int, lr: float=0.99, lr_decay: float=0.8,  beta1: float=0.99, beta2: float=0.01, penalty: float=1.0):
        self.lr = lr
        self.lr_decay = lr_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.penalty = penalty
        self.feedforward_potential = torch.zeros(in_dim, out_dim)
        self.recurrent_potential = torch.zeros(out_dim, out_dim)
        self.dropout = torch.tensor([True]*out_dim)
        self.counter = 1
        
    def step(self, output: torch.Tensor, input: torch.Tensor=torch.tensor([]), normalized: bool=False, use_dropout: bool=False, use_transform: bool=False, use_floor: bool=False):
        """Function to transform the layer activity and update potentials
            Args:
                output: 1D tensor of current layer activity
                input: Crresponding input tensor to the layer
                normalized: A boolean specifying normalization of layer acitivity
                use_dropout: A boolean specifying using dropouts
                use_transform: A boolean specifying a specific non-linear transformation of the layer activity
                use_floor: A boolean specifying if the potential values need to be floored to the nearest integer values
            Return:
                Nothing 
        """
        output = self._filter(output) if use_dropout else output
        output = self._normalize(output) if normalized else output
        output = self._transform(output) if use_transform else output

        self.recurrent_potential = self.beta1*self.recurrent_potential + self.beta2*self._recurrent_potential(output, use_floor, self.penalty)
        if input.numel() != 0:
            self.feedforward_potential = self.beta1*self.feedforward_potential + self.beta2*self._feedforward_potential(output, input)

    def organize(self, weights: torch.Tensor=torch.tensor([])):
        """Function to produce updated weights based on potentials and previous weights
            Args:
                weights: (optional) previous weights
            Return:
                updated_weights: 2D tensor of updated weights
        """
        if weights.numel() == 0:
            updated_weights = (self.recurrent_potential > 0).float()
            #self.feedforward_potential.fill_(0.0)
            self.recurrent_potential.fill_(0.0)
        else:
            correction_factor = 1/(1 - self.beta1**self.counter)
            updated_weights = (1 - self.lr)*weights + correction_factor*self.lr*(self.feedforward_potential - torch.mm(weights, self.recurrent_potential))
            self.counter += 1
            self.lr = self.lr_decay*self.lr
        
        return updated_weights

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
    def _feedforward_potential(output: torch.Tensor, input: torch.Tensor):
        """Function to caluclate the hebbian potentials based on coactivities
            Args:
                output: 1D tensor of current layer activity
                input: Crresponding input tensor to the layer
            Return:
                Hebbian connection potentials
        """
        return torch.mm(input.T, output)
    
    @staticmethod
    def _recurrent_potential(output: torch.Tensor, use_floor: bool, penalty: float):
        """Function to caluclate the hebbian potentials based on coactivities
            Args:
                output: 1D tensor of current layer activity
                use_floor: A boolean specifying if the potentials need to be floored to the nearest integer
                penalty: A float value specifying the relative penalty in case of activation mismatch
            Return:
                Hebbian connection potentials
        """
        potential = torch.mm(output.T, output)
        if use_floor:
            #penalty = 1 - torch.exp(torch.tensor(-penalty*0.5)).item()
            potential = torch.floor(potential)
            potential = (potential >= 0)*potential + penalty*(potential < 0)*potential
        
        return potential

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