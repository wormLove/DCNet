import torch
from torch.nn.functional import normalize
from torch.linalg import multi_dot

def randomizer_matrix(m: int, n: int):
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
        
def effective_dimensionality(S: torch.Tensor):
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
    
def initialize_connections(sample_data: torch.Tensor, output_dims: int):
    """Function to obtain the initial connectivity tensor given a sample of data
        Args:
            sample_data: A tensor of sample data of dimensions nsamples by data_dim
            output_dim: An integer specifying the dimesion of the network output
        Return:
            A normalized tensor of initial connectivity
    """
    U, S, V = torch.pca_lowrank(sample_data, q=sample_data.shape[1], center=True)
    Sn, n = effective_dimensionality(S)
    Sn_inv = torch.diag(Sn[:n]**(-1))
    randomizer = randomizer_matrix(output_dims, n)
    init_connections = multi_dot((V[:,:n], Sn_inv, randomizer.T))
    return normalize(init_connections, p=2, dim=0)

