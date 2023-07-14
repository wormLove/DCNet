import torch
from torch import nn
from torch.linalg import matrix_rank, multi_dot
from torch.nn.functional import normalize

class LayerThresholding(nn.Module):
    def __init__(self, alpha: float=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, input: torch.Tensor):
        return input*(input > self.alpha*torch.std(input).item())

class Feedforward(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        assert weights.dim() == 2, "connectivity matrix must be 2D"
        self.weights = normalize(weights, p=2, dim=0)

    def forward(self, input: torch.Tensor):
        return torch.mm(input, self.weights)
    
    def update(self, weights: torch.Tensor, unit_norm: bool=True):
        self.weights =  normalize(weights, p=2, dim=0) if unit_norm else weights
    


class Recurrent(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        assert weights.dim() == 2 and weights.shape[0] == weights.shape[1], "recurrent connections must be all to all and 2D"
        self.weights = weights

    def forward(self, input: torch.Tensor):
        approx_recurrence = self.approximate_recurrence(input)
        return torch.mm(input, approx_recurrence)
    
    def update(self, weights: torch.Tensor):
        self.weights =  weights
    
    def approximate_recurrence(self, input: torch.Tensor):
        # check shape constrains and initialize selection variables
        m = input.shape[1]
        selection_diag = torch.zeros(m)
        selection_indx = torch.flatten(torch.argsort(input, descending=True), start_dim=0)
        
        # perform binary search for the best selction diagonal
        left, right = 0, m
        while left < right:
            mid = int(0.5*(left+right))
            selection_diag[selection_indx[:mid+1]] = 1
            sys_mat = self.system_matrix(selection_diag)
            if matrix_rank(sys_mat) < m:
                right = mid
            else:
                left = mid+1
            selection_diag[torch.nonzero(selection_diag)] = 0
        selection_diag[selection_indx[:left]] = 1 
        
        # return the best selection diagonal
        return torch.inverse(self.system_matrix(selection_diag))
    
    def system_matrix(self, selection_diag: torch.Tensor):
        return torch.eye(len(selection_diag)) + torch.mm(torch.diag(selection_diag), self.weights) - torch.diag(selection_diag)
    


class DiscriminationModule(nn.Module):
    def __init__(self, weights: torch.Tensor, lr: float=0.99, lr_decay: float=0.8, beta: float=0.99, alpha: float=1.0):
        super().__init__()
        self.feedforward = Feedforward(weights)
        self.recurrent = Recurrent(self.recurrent_weights())
        self.activation = LayerThresholding(alpha=alpha)
        self.lr = lr
        self.lr_decay = lr_decay
        self.beta = beta
        self.velocity_hebb = torch.zeros(weights.shape[0], weights.shape[1])
        self.velocity_anti_hebb = torch.zeros(weights.shape[1], weights.shape[1])
        self.dropout = torch.zeros(1, weights.shape[1]).bool()
        self.counter = 1
        

    def forward(self, input: torch.Tensor):
        assert input.dim() == 2 and input.shape[0] == 1, "input must be a row vector"
        out_ = self.feedforward(input)
        out_ = self.recurrent(out_)
        out_f = self.activation(out_)
        self.update(input, out_f)
        return out_f

    def recurrent_weights(self): 
        return torch.mm(self.feedforward.weights.T, self.feedforward.weights)
    
    def update(self, input: torch.Tensor, out_f: torch.Tensor):
        # droput the previously active cells
        out_filtered = torch.logical_not(self.dropout)*out_f
        norm_out_filtered = torch.norm(out_filtered, p='fro').item()
        
        # update potentials if response is non-zero
        if norm_out_filtered:
            self.velocity_hebb = self.beta*self.velocity_hebb + (1-self.beta)*torch.mm(input.T, out_filtered)/norm_out_filtered
            self.velocity_anti_hebb = self.beta*self.velocity_anti_hebb + (1-self.beta)*torch.mm(out_filtered.T, out_filtered)/norm_out_filtered**2
        
        # update droputs to be the active cells
        self.dropout = out_f > 0


    def organize(self):
        # calculate the updated weights
        correction_factor = 1/(1 - self.beta**self.counter)
        updated_weights = (1 - self.lr)*self.feedforward.weights + correction_factor*self.lr*(self.velocity_hebb - torch.mm(self.feedforward.weights, self.velocity_anti_hebb))

        # update weights in feedforward and recurrent connections 
        self.feedforward.update(updated_weights)
        self.recurrent.update(self.recurrent_weights())
        
        # update learning rate and organization counter
        self.counter += 1
        self.lr = self.lr_decay*self.lr

    def connections(self):
        return torch.clone(self.feedforward.weights)



class ClassificationModule(nn.Module):
    def __init__(self, weights: torch.Tensor, alpha: float=1.0, penalty: float=1.0):
        super().__init__()
        self.feedforward1 = Feedforward(weights)
        self.feedforward2 = Feedforward(torch.eye(weights.shape[1]))
        self.potential = torch.zeros(weights.shape[1], weights.shape[1])
        self.activation = LayerThresholding(alpha=alpha)
        self.penalty = penalty

    def forward(self, input: torch.Tensor):
        assert input.dim() == 2 and input.shape[0] == 1, "input must be a row vector"
        out_ = self.feedforward1(input)
        out_ = self.feedforward2(out_)
        out_f = self.activation(out_)
        self.update(out_f)
        return out_f
    
    def transform(self, out_f: torch.Tensor):
        out_f_transformed = -0.5*torch.ones(out_f.shape)
        out_f_transformed[out_f > 0] = 1.0
        return out_f_transformed
    
    def update(self, out_f: torch.Tensor):
        out_f_transformed = self.transform(out_f)
        update_matrix = torch.floor(torch.mm(out_f_transformed.T, out_f_transformed))
        update_matrix = (update_matrix >= 0)*update_matrix + self.penalty*(update_matrix < 0)*update_matrix
        self.potential += update_matrix
    
    def organize(self):
        excitatory_weights = (self.potential > 0).float()
        self.feedforward2.update(excitatory_weights, unit_norm=False)
        self.potential[:,:] = 0.0

    def connections(self):
        return torch.clone(self.feedforward2.weights)
    
    