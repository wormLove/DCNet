import torch
import torch.nn.functional as F
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.linalg import multi_dot
from abc import ABC, abstractmethod

#from loading import SequentialLoader, RandomLoader
from transformations import Transform

class Initializer(ABC):
    """A template class to form different types of initializers
    """
    @abstractmethod
    def weights(self):
        """Returns initial weights tensor
        """
        pass
    
    @property
    @abstractmethod
    def in_dim(self):
        """Returns input dimensions to a layer
        """
        pass


class DatasetInitializer(Initializer):
    """Class to initialize dataset specicfic initial connection and a data loader
        Values:
            dataset: A dataset object
            transforms: A list of transformations applied to the inputs in sequence
            split: A float value specifying the fraction of dataset to use in intialization
    """
    def __init__(self, dataset: Dataset, transforms: Transform, split: float = 0.25):
        self.dataset = dataset
        self.transforms = transforms
        self.split = split

    def weights(self, out_dim: int):
        """Args: out_dim: An integer specifying the output dimensions of the layer
        """
        # get sample data
        batch_size = int(self.split*len(self.dataset)) # type: ignore
        sample_data = self._sample_data(batch_size)
            
        # perform svd to get prinicpal axes
        U, S, V = torch.pca_lowrank(sample_data, q=sample_data.shape[1], center=True)
        Sn, n = self._effective_dims(S)
        Sn_inv = torch.diag(Sn[:n]**(-1))
            
        # randomize the prinicpal axes
        randomizer = self._randomizer(out_dim, n)
        connections = multi_dot((V[:,:n], Sn_inv, randomizer.T))
        return connections
    
    @property
    def in_dim(self):
        """Returns input dimensions
        """
        sample, target = next(iter(self.dataset))
        return self.transforms(sample, label=target).shape[1]
    
    def _sample_data(self, batch_size: int):
        """Function to return sample data with optional padded one-hot labels
            Args:
                batch_size: An integer specifying the sample data size
            Return:
                Sample data tensor
        """
        batch, targets = next(iter(DataLoader(self.dataset, sampler=RandomSampler(self.dataset), batch_size=batch_size)))
        sample_data = torch.empty(batch_size, self.in_dim)
        for idx, sample in enumerate(batch):
            sample_data[idx,:] = self.transforms(sample, label=targets[idx].item())
        return sample_data

    @staticmethod
    def _randomizer(m: int, n: int):
        """Function to generate a near-orthogonal tensor to randomize connections
            Args:
                m: an integer specifying the output dimensions
                n: an integer specifying the input dimensions
                note: m >= n
            Return:
                A near-orthogonal tensor of dimensions m by n
        """
        assert m >= n, f"{m=} should be greater than or equal to {n=}"
        _, _, V = torch.pca_lowrank(torch.rand(n, n), q=n)
        if m > n:
            random_matrix = normalize(torch.randn(m-n, n), p=2, dim=1)
            Vc = torch.mm(random_matrix, V)
            return torch.vstack((V, Vc))
        else:
            return V
    
    @staticmethod
    def _effective_dims(S: torch.Tensor):
        """Function to calculate the effective dimensionality of the sample data
            Args:
                S: A vector of singular values
            Return:
                A tuple containg a 1D tensor of normalized singular values and an integer value of the effective dimensionality
        """
        assert S.dim() == 1, "singular values must be in 1D tensor"
        Sn = S**2/torch.sum(S**2)
        n = 0
        for i in range(len(Sn)):
            if torch.sum(Sn[:i]) > 0.95:
                n = i+1
                break
        return Sn, n
    

class IdentityInitializer(Initializer):
    """Class to initialize initial connections as identity tensor
    """ 
    def __init__(self):
        self._in_dim = 0
    
    def weights(self, out_dim: int):
        """Args: out_dim: An integer specifying the output dimensions of the layer
        """
        self._in_dim = out_dim
        return torch.eye(out_dim)
    
    @property
    def in_dim(self):
        """Returns input dimensions
        """
        return self._in_dim  


class RandomInitializer(Initializer):
    """Class to initialize initial connections as random tensor
        Values:
            in_dim: (Optional) An integer values specifying the input dimensions when dataset is not specified
            dataset: (Optional) A dataset object
            num_classes: (Optional) An integer specifying number of classes in the dataset
            (Note: One of the dataset or the in_dim must be initialized)
    """
    def __init__(self, in_dim: int=0, dataset: Dataset = None, transforms: Transform = None):
        assert (dataset is not None) ^ (in_dim != 0), "either the dataset or the input dimensions must be specified but not both"
        self._in_dim = in_dim
        self.dataset = dataset
        self.transforms = transforms
        
    def weights(self, out_dim: int):
        """Args: out_dim: An integer specifying the output dimensions of the layer
        """
        return torch.randn(self.in_dim, out_dim)
    
    @property
    def in_dim(self):
        """Returns input dimensions
        """
        if self.dataset is not None:
            sample, target = next(iter(self.dataset))
            return self.transforms(sample, label=target).shape[1]
        return self._in_dim 

    
class StateInitializer(Initializer):
    """Class to initialize initial connections as some predefined tensor and a data loader if dataset is specified
        Values:
            _weights: A predefined weight tensor 
    """
    def __init__(self, weights: torch.Tensor):
        assert weights.dim() == 2, "weights tensor must be 2D"
        self._weights = weights
        
    def weights(self, out_dim: int):
        assert out_dim == self._weights.shape[1], "layer dimensions and weights tensor are incompatible"
        return self._weights
    
    @property
    def in_dim(self):
        """Returns input dimensions
        """
        return self._weights.shape[0]
