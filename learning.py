import torch
from abc import ABC, abstractmethod
from typing import Dict
from torch.nn.functional import normalize, one_hot

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
        self.margin = kwargs.get('margin', 1.0)
        self.threshold = kwargs.get('threshold', 0.2)
        
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
        
    def organize(self, weights: torch.Tensor):
        """Function to produce updated weights based on potentials
            Return:
                updated_weights: 2D tensor of updated weights
        """
        updated_weights = weights.logical_or(self.potential > 0).float()
        self.potential.fill_(0.0)
        return updated_weights

    
    @staticmethod
    def _potential(x: torch.Tensor):
        return torch.floor(torch.mm(x.T, x))

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
    
class AdaptationOrganizer(Organizer):
    def __init__(self, out_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.potential = torch.zeros(out_dim, out_dim)
        self.input_embeddings = torch.empty(0)
        self.output_embeddings = torch.empty(0)
        self._loss = 0.0
    
    
    def step(self, input: torch.Tensor, output: torch.Tensor):
        self._accumulate(input, output)
        try:
            anc, pos, neg = self.input_embeddings[0].unsqueeze(0), self.input_embeddings[1].unsqueeze(0), self.input_embeddings[2].unsqueeze(0)
            if self._current_loss() > 0:
                filter_pos, filter_neg = self.filters
                self.potential = self.beta*self.potential + (1-self.beta)*(filter_pos*self._potential(anc, pos) - filter_neg*self._potential(anc, neg))
            self._dump()
        except IndexError:
            pass
            
    def organize(self, weights: torch.Tensor):
        updated_weights = weights + self.lr*torch.mm(self.potential, weights)
        updated_weights = normalize(updated_weights, p=2, dim=0)
        self._loss = 0.0
        return updated_weights
    
    @staticmethod
    def _potential(x: torch.Tensor, y: torch.Tensor):
        return torch.mm(x.T, y) + torch.mm(y.T, x)
    
    def _accumulate(self, input: torch.Tensor, output: torch.Tensor):
        self.input_embeddings = torch.cat((self.input_embeddings, self._normalize(input, output)), dim=0)
        self.output_embeddings = torch.cat((self.output_embeddings, self._normalize(output, output)), dim=0)
    
    def _dump(self):
        self.input_embeddings = torch.empty(0)
        self.output_embeddings = torch.empty(0)
        
    def _current_loss(self):
        emb_anc, emb_pos, emb_neg = self.output_embeddings[0].unsqueeze(0), self.output_embeddings[1].unsqueeze(0), self.output_embeddings[2].unsqueeze(0)
        d_pos = 1 - torch.mm(emb_anc, emb_pos.T).item()
        d_neg = 1 - torch.mm(emb_anc, emb_neg.T).item()
        current_loss = d_pos - d_neg + self.margin
        self._loss += current_loss
        return current_loss
    
    @staticmethod
    def _normalize(x: torch.Tensor, y: torch.Tensor):
        """Function to normalize layer activity wrt other activity
            Args:
                x: 1D tensor of 1st instance of activity
                y: 1D tensor of 2nd instance of activity
            Return:
                Normalized activity
        """
        y_norm = torch.norm(y, p='fro').item()
        x = x/y_norm if y_norm != 0 else torch.zeros_like(x)
        return x
    
    @property
    def filters(self):
        filter_pos = (self.output_embeddings[0].unsqueeze(0) > 0).logical_and(self.output_embeddings[1].unsqueeze(0) > 0)
        filter_neg = (self.output_embeddings[0].unsqueeze(0) > 0).logical_and(self.output_embeddings[2].unsqueeze(0) > 0)
        return filter_pos, filter_neg
    
    @property
    def loss(self):
        return self._loss
    

class AdaptationOrganizer_u(Organizer):
    def __init__(self, out_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.potential = torch.zeros(out_dim, out_dim)
        self.input_last = torch.zeros(1, out_dim)
        self.output_last = torch.zeros(1, out_dim)
        self._loss = 0.0
    
    
    def step(self, input: torch.Tensor, output: torch.Tensor):
        anc = self._normalize(self.input_last, self.output_last)
        exm = self._normalize(input, output)
        sign = self._sign(input)
        if self._current_loss(output, sign) > 0:
            self.potential = self.beta*self.potential + sign*(1-self.beta)*self._filter(output)*self._potential(anc, exm)
        self._collect(input, output)

            
    def organize(self, weights: torch.Tensor):
        updated_weights = weights + self.lr*torch.mm(self.potential, weights)
        updated_weights = normalize(updated_weights, p=2, dim=0)
        self._loss = 0.0
        return updated_weights
    
    @staticmethod
    def _potential(x: torch.Tensor, y: torch.Tensor):
        return torch.mm(x.T, y) + torch.mm(y.T, x)
    
    def _collect(self, input: torch.Tensor, output: torch.Tensor):
        self.input_last = input
        self.output_last = output
    
    def _sign(self, input: torch.Tensor):
        intersection = (self.input_last > 0).logical_and(input > 0)
        union = min((self.input_last > 0).sum().item(), (input > 0).sum().item())
        similarity = intersection.sum().item()/(union+1)
        return 1*(similarity > self.threshold) - 1*(similarity <= self.threshold)
        
    def _current_loss(self, output: torch.Tensor, sign: int):
        anc = self._normalize(self.output_last, self.output_last)
        exm = self._normalize(output, output)
        d = 1 - torch.mm(anc, exm.T).item()
        sign = 0.5*(sign+1)
        current_loss = sign*d + (1-sign)*(self.margin-d)
        self._loss += current_loss
        return current_loss
    
    @staticmethod
    def _normalize(x: torch.Tensor, y: torch.Tensor):
        """Function to normalize layer activity wrt other activity
            Args:
                x: 1D tensor of 1st instance of activity
                y: 1D tensor of 2nd instance of activity
            Return:
                Normalized activity
        """
        y_norm = torch.norm(y, p='fro').item()
        x = x/y_norm if y_norm != 0 else torch.zeros_like(x)
        return x
    
    def _filter(self, output: torch.Tensor):
        return (self.output_last > 0).logical_and(output > 0)
    
    @property
    def loss(self):
        return self._loss
    
class AnchorOrganizer(Organizer):
    def __init__(self, out_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.potential = torch.zeros(out_dim, out_dim)
        self.anchors_inp = torch.empty(0)
        self.anchors_out = torch.empty(0)
        self.mean_rep = torch.zeros(1, out_dim)
        #self.norms = []
        #self.input_embeddings = torch.empty(0)
        #self.output_embeddings = torch.empty(0)
        self._loss = 0.0
    
    
    def step(self, input: torch.Tensor, output: torch.Tensor):
        #self._accumulate(input, output)
        #try:
        #    anc, pos, neg = self.input_embeddings[0].unsqueeze(0), self.input_embeddings[1].unsqueeze(0), self.input_embeddings[2].unsqueeze(0)
        #    if self._current_loss() > 0:
        #        filter_pos, filter_neg = self.filters
        #        self.potential = self.beta*self.potential + (1-self.beta)*(filter_pos*self._potential(anc, pos) - filter_neg*self._potential(anc, neg))
        #    self._dump()
        #except IndexError:
        #    pass
        #pos_idxs, neg_idxs, similarities = self._indexes(input)
        self.current_input, self.current_output = input, output
        self.mean_rep = self.beta*self.mean_rep + (1 - self.beta)*input
        pos_potential, neg_potential = self._potentials()
        self.potential = self.beta*self.potential + (1-self.beta)*(neg_potential - pos_potential)
            
    def organize(self, weights: torch.Tensor):
        updated_weights = weights + self.lr*torch.mm(self.potential, weights)
        #updated_weights = normalize(updated_weights, p=2, dim=0)
        self._loss = updated_weights.sum()
        self._dump()
        #self._loss = 0.0
        return updated_weights
    
    @staticmethod
    def _potential(x: torch.Tensor, y: torch.Tensor):
        return torch.mm(x.T, y) + torch.mm(y.T, x)
    
    #def _accumulate(self, input: torch.Tensor, output: torch.Tensor):
    #    self.input_embeddings = torch.cat((self.input_embeddings, self._normalize(input, output)), dim=0)
    #    self.output_embeddings = torch.cat((self.output_embeddings, self._normalize(output, output)), dim=0)
        
    def _potentials(self):
        similarities = self._similarity()
        threshold = 0.5*(1 + self.margin)
        pos_potential, neg_potential = torch.zeros_like(self.potential), torch.zeros_like(self.potential)
        if similarities.nelement() > 0:
            #threshold = 0.5*(1 + self.margin)
            #pos_idxs = (similarities.flatten() > threshold).nonzero().squeeze(1)
            #neg_idxs = (similarities.flatten() <= threshold).nonzero().squeeze(1)
            #pos_idxs = similarities.argmax().unsqueeze(0).tolist()
            #neg_idxs = list(set([*range(similarities.shape[1])]) - set(pos_idxs))
            #self._accumulate(pos_potential, pos_idxs)
            #self._accumulate(neg_potential, neg_idxs) if len(neg_idxs) > 0 else None
            pos_idxs = (similarities.flatten() > threshold).nonzero().squeeze(1).tolist()
            if len(pos_idxs) > 0:
                neg_idxs = list(set(range(similarities.shape[1])) - set(pos_idxs))
                self._accumulate(pos_potential, pos_idxs)
                self._accumulate(neg_potential, neg_idxs) if len(neg_idxs) > 0 else None
        self._update_anchors(similarities, threshold)
            
        return pos_potential, neg_potential
            
    
    def _accumulate(self, potential: torch.Tensor, idxs: torch.Tensor):
        for idx in idxs:
            inp = self._normalize(self.current_input, self.current_output)
            anc = self._normalize(self.anchors_inp[idx].unsqueeze(0), self.anchors_out[idx].unsqueeze(0))
            filter = self.filter(self.current_output, self.anchors_out[idx].unsqueeze(0))
            potential += filter*self._potential(inp, anc)
        potential /= len(idxs)

    
    def _dump(self):
        self.anchors_inp = torch.empty(0)
        self.anchors_out = torch.empty(0)
        
    #def _current_loss(self):
    #    emb_anc, emb_pos, emb_neg = self.output_embeddings[0].unsqueeze(0), self.output_embeddings[1].unsqueeze(0), self.output_embeddings[2].unsqueeze(0)
    #    d_pos = 1 - torch.mm(emb_anc, emb_pos.T).item()
    #    d_neg = 1 - torch.mm(emb_anc, emb_neg.T).item()
    #    current_loss = d_pos - d_neg + self.margin
    #    self._loss += current_loss
    #    return current_loss
    
    @staticmethod
    def _normalize(x: torch.Tensor, y: torch.Tensor):
        """Function to normalize layer activity wrt other activity
            Args:
                x: 1D tensor of 1st instance of activity
                y: 1D tensor of 2nd instance of activity
            Return:
                Normalized activity
        """
        y_norm = torch.norm(y, p='fro').item()
        x = x/y_norm if y_norm != 0 else torch.zeros_like(x)
        return x
    
    #@property
    #def filters(self):
    #    filter_pos = (self.output_embeddings[0].unsqueeze(0) > 0).logical_and(self.output_embeddings[1].unsqueeze(0) > 0)
    #    filter_neg = (self.output_embeddings[0].unsqueeze(0) > 0).logical_and(self.output_embeddings[2].unsqueeze(0) > 0)
    #    return filter_pos, filter_neg
    
    @property
    def nanchors(self):
        return self.anchors_inp.shape[0]
    
    @property
    def loss(self):
        return self._loss
        
    
    @staticmethod
    def filter(x: torch.Tensor, y: torch.Tensor):
        return (x > 0).logical_and(y > 0)
    
    def _update_anchors(self, similarities: torch.Tensor, threshold: float):
        #threshold_u, threshold_l = 0.5*(1 + self.margin), 0.5*(1 - self.margin)
        if similarities.nelement() == 0:
            self.anchors_inp = torch.cat((self.anchors_inp, self.current_input), dim=0)
            self.anchors_out = torch.cat((self.anchors_out, self.current_output), dim=0)
        else:
            pos_idxs = (similarities.flatten() > threshold).nonzero().squeeze(1)
            if pos_idxs.nelement() > 0:
                for idx in pos_idxs:
                    if self._updatable(idx.item()):
                        #self.anchors_inp[idx.item()] = 0.5*(self.anchors_inp[idx.item()] + self.current_input.squeeze())
                        #self.anchors_out[idx.item()] = 0.5*(self.anchors_out[idx.item()] + self.current_output.squeeze())
                        self.anchors_inp[idx.item()] = self.current_input
                        self.anchors_out[idx.item()] = self.current_output
            elif all(similarities.flatten() < 1-threshold):
                self.anchors_inp = torch.cat((self.anchors_inp, self.current_input), dim=0)
                self.anchors_out = torch.cat((self.anchors_out, self.current_output), dim=0)        
            
            
    def _updatable(self, idx: int):
        #anchors_normalized = normalize(self.anchors_inp, p=2, dim=1)
        #new_anchor = 0.5*(self.anchors_inp[idx].unsqueeze(0) + self.current_input)
        #new_anchor_normalized = normalize(new_anchor, p=2, dim=1)
        #new_similarities = torch.mm(new_anchor_normalized, anchors_normalized.T)
        #return all(new_similarities.flatten() < 1-threshold)
        distance_prev = torch.norm(self.anchors_inp[idx] - self.mean_rep, p='fro')
        distance_curr = torch.norm(self.current_input - self.mean_rep, p='fro')
        return distance_prev < distance_curr
        

    
    def _similarity(self):
        input_normalized = normalize(self.current_input, p=2, dim=1)
        try:
            anchors_normalized = normalize(self.anchors_inp, p=2, dim=1)
            similarities = torch.mm(input_normalized, anchors_normalized.T)
        except IndexError:
            similarities = torch.empty(0)
        return similarities
    
class AttentionOrganizer(Organizer):
    def __init__(self, out_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.keys_ = torch.empty(0)
        self.values_ = torch.empty(0)
        self.out_dim = out_dim
        
    
    def step(self, inputs: torch.Tensor, targets: torch.Tensor):
        targets = targets.long() if targets.nelement() > 0 else None
        self._initialize_keys_values(inputs, targets) if self.keys_.nelement() == 0 else self._update_keys_values(inputs, targets)
            
        
    
    def organize(self):
        self.keys_ = torch.empty(0)
        self.values_ = torch.empty(0)
        
    @property
    def _potential(self):
        pass
    
    @property
    def keys(self):
        return self.keys_ 
    
    @property
    def values(self):
        return self.values_
    
    
    def _updatable(self, input: torch.Tensor):
        input_normalized = normalize(input, p=2, dim=1)
        keys_normalized = normalize(self.keys_, p=2, dim=1) if self.keys_.nelement() > 0 else torch.zeros_like(input)
        
        similarities = torch.mm(input_normalized, keys_normalized.T)
        return all(similarities.flatten() < self.threshold)
    
    def _indexes(self, inputs: torch.Tensor):
        input_normalized = normalize(inputs, p=2, dim=1)
        keys_normalized = normalize(self.keys_, p=2, dim=1)
        
        similarities = torch.mm(input_normalized, keys_normalized.T)
        indexes = ((similarities < self.threshold).sum(dim=1) == len(self.keys_)).nonzero().squeeze(1)
        return indexes
    
    def _initialize_keys_values(self, inputs: torch.Tensor, targets: torch.Tensor):
        for idx, input in enumerate(inputs):
            idx = torch.tensor(idx).unsqueeze(0)
            if self._updatable(input.unsqueeze(0)):
                self.keys_ = torch.cat((self.keys_, input.unsqueeze(0)), dim=0)
                self.values_ = torch.cat((self.values_, one_hot(targets[idx], num_classes=self.out_dim)), dim=0) if targets is not None else torch.empty(0)
                
    def _update_keys_values(self, inputs: torch.Tensor, targets: torch.Tensor):
        idxs = self._indexes(inputs)
        self.keys_ = torch.cat((self.keys_, inputs[idxs]), dim=0)
        self.values_ = torch.cat((self.values_, one_hot(targets[idxs], num_classes=self.out_dim)), dim=0) if targets is not None else torch.empty(0)