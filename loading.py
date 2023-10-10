import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torch.nn.functional as F
from transformations import Transform

class RandomLoader:
    """A loader class to load sample from dataset randomly
        Values:
            dataset: A dataset object
            transforms: A list of transformations applied to the inputs in sequence
    """
    def __init__(self, dataset: Dataset, transforms: Transform):
        self.dataset = dataset
        self.transforms = transforms
    
    def __call__(self, nsamples: int):
        self.nsamples = nsamples
        self._initialize_loader()
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.nsamples:
            self.current += 1
            sample, target = next(iter(self.loader))
            return self.transforms(sample, label=target.item())
        raise StopIteration
    
    def _initialize_loader(self):
        self.loader = DataLoader(self.dataset, sampler=RandomSampler(self.dataset))
        self.current = 0


class SequentialLoader:
    """A loader class to load sample from dataset in ordered sequence
        Values:
            dataset: A dataset object
            transforms: A list of transformations applied to the inputs in sequence
    """
    def __init__(self, dataset: Dataset, transforms: Transform):
        self.dataset = dataset
        self.transforms = transforms
    
    def __call__(self, nsamples: int):
        self.nsamples = nsamples
        self._initialize_batch()
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.nsamples:
            self.current += 1
            return self.transforms(self.batch[self.current-1], label=self.targets[self.current-1].item())
        raise StopIteration
    
    def _initialize_batch(self):
        """Function to initialize an ordered batch
        """
        self.batch, self.targets = self._ordered_batch()
        self.current = 0
        
    def _ordered_batch(self):
        """Generates a random batch and orders it
        Returns:
            A tuple of ordered sample and target tensors
        """
        batch, targets = next(iter(DataLoader(self.dataset, sampler=RandomSampler(self.dataset), batch_size=self.nsamples)))
        sequence = self._sequence(targets)
        return batch[sequence], targets[sequence]
    
    @staticmethod
    def _sequence(targets: torch.Tensor):
        """A method to ordere the targets of the same label together
            Args:
                targets: A 1D tensor of targets
            Returns:
                ordered list of target indices
        """
        loading_sequence = []
        remaining_targets = targets.unique().tolist()
        for target in targets.tolist():
            if target in remaining_targets:
                target_indxs = (targets == target).nonzero()
                loading_sequence.append(target_indxs.flatten().tolist())
                remaining_targets.remove(target)
            if not remaining_targets:
                break
        loading_sequence = [idx for lst in loading_sequence for idx in lst]
        return loading_sequence


class TripletLoader:
    """A loader class to load sample from dataset in triplets of anchor, positive, and negative samples
        Values:
            dataset: A dataset object
            transforms: A list of transformations applied to the inputs in sequence
    """
    def __init__(self, dataset: Dataset, transforms: Transform):
        self.dataset = dataset
        self.transforms = transforms
        
    def __call__(self, nsamples: int):
        self.nsamples = nsamples
        self._initialize_loader()
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.nsamples:
            transformed_sample = self.transforms(self.triplet[0][self.ret_idx], label=self.triplet[1][self.ret_idx].item())
            self.ret_idx += 1
            if self.ret_idx == 3:
                self.triplet = self._triplet()
                self.ret_idx = 0
                self.current += 1
            return transformed_sample  
        raise StopIteration
    
    def _initialize_loader(self):
        """Function to initialize the loader
        """
        self.loader = DataLoader(self.dataset, sampler=RandomSampler(self.dataset), batch_size=100)
        self.triplet = self._triplet()
        self.current = 0
        self.ret_idx = 0
        
    def _triplet(self):
        """Function to generate triplets
        Returns:
            triplets of samples and repective targets
        """
        anchor_idx, positive_idx, negative_idx = 0, float('inf'), float('inf')
        s, t = next(iter(self.loader))
        while torch.tensor(positive_idx).isinf() or torch.tensor(negative_idx).isinf():
            try:
                positive_idx = (t == t[0]).nonzero().flatten()[1].item()
                negative_idx = (t != t[0]).nonzero().flatten()[1].item()
            except IndexError:
                s, t = next(iter(self.loader))
        indices = [anchor_idx, positive_idx, negative_idx]
        return s[indices], t[indices]