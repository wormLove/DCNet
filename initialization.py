import torch
import torch.nn.functional as F
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.linalg import multi_dot

class Loader:
    """An iterable class to return data samples with optional padding of one-hot label vectors
    """
    def __init__(self, loader: DataLoader, nsamples: int, num_classes: int, pad_labels: bool=True):
        self.loader = loader
        self.nsamples = nsamples
        self.num_classes = num_classes
        self.pad_labels = pad_labels
        self.current = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.nsamples:
            self.current += 1
            sample, target = next(iter(self.loader))
            pad = F.one_hot(target, num_classes=self.num_classes) if self.pad_labels else torch.zeros(1, self.num_classes)
            return torch.cat((sample.reshape(1, -1), pad), dim=1)
        raise StopIteration


class Initializer:
    """Class to initialize dataset specicfic initial connection and a data loader
    """
    def __init__(self, dataset: Dataset, num_classes: int, pad_labels: bool=True):
        self.dataset = dataset
        self.num_classes = num_classes
        self.pad_labels = pad_labels
    
    def weights(self, out_dim: int, split: int):
        """Function to obtain the initial connectivity tensor for the given dataset
            Args:
                out_dim: An integer specifying the dimesion of the network output
                split: An integer specifying the sample data size for initializing connections 
            Return:
                A normalized tensor of initial connectivity
        """
        # get sample data
        sample_data = self._sample_data(split, self.pad_labels)
        
        # perform svd to get prinicpal axes
        U, S, V = torch.pca_lowrank(sample_data, q=sample_data.shape[1], center=True)
        Sn, n = self._effective_dims(S)
        Sn_inv = torch.diag(Sn[:n]**(-1))
        
        # randomize the prinicpal axes
        randomizer = self._randomizer(out_dim, n)
        init_connections = multi_dot((V[:,:n], Sn_inv, randomizer.T))
        return normalize(init_connections, p=2, dim=0)
    
    def loader(self, nsamples: int):
        """Function to generate a data loader which returns flattened examples with optional padded one-hot labels in batch size 1 
            Args:
                nsamples: An integer value specifying the number of data points to load
            Return:
                DataLoader
        """
        loader = DataLoader(self.dataset, sampler=RandomSampler(self.dataset)) # type: ignore
        return Loader(loader, nsamples, self.num_classes, self.pad_labels)
    
    def _sample_data(self, split: int, pad_labels: bool):
        """Function to return sample data with optional padded one-hot labels
            Args:
                split: An integer specifying the sample data size
                pad_labels: A bool specifying whether to pad one-hot label vectors
            Return:
                Sample data tensor
        """
        loader = DataLoader(self.dataset, sampler=RandomSampler(self.dataset), batch_size=split) # type: ignore
        samples, targets = next(iter(loader))
        pad = F.one_hot(targets, num_classes=self.num_classes) if pad_labels else torch.zeros(split, self.num_classes)
        return torch.cat((samples.reshape(split, -1), pad), dim=1)
    
    def _randomizer(self, m: int, n: int):
            """Function to generate a near-orthogonal tensor to randomize connections
                Args:
                    m: an integer specifying the output dimensions
                    n: an integer specifying the input dimensions
                    note: m >= n
                Return:
                    A near-orthogonal tensor of dimensions m by n
            """
            assert m >= n, f"m={m} should be greater than or equal to n={n}"
            _, _, V = torch.pca_lowrank(torch.rand(n, n), q=n)
            if m > n:
                random_matrix = normalize(torch.randn(m-n, n), p=2, dim=1)
                Vc = torch.mm(random_matrix, V)
                return torch.vstack((V, Vc))
            else:
                return V
    
    def _effective_dims(self, S: torch.Tensor):
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

