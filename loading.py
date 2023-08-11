import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torch.nn.functional as F

class RandomLoader:
    """A loader class to load sample from dataset randomly
        Values:
            dataset: A dataset object
            num_classes: Number of unique classes in the dataset
    """
    def __init__(self, dataset: Dataset, num_classes: int):
        self.dataset = dataset
        self.num_classes = num_classes
        
        
    def __call__(self, nsamples: int, pad_label: bool=True):
        self.nsamples = nsamples
        self.pad_labels = pad_label
        self._initialize_loader()
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.nsamples:
            self.current += 1
            sample, target = next(iter(self.loader))
            pad = F.one_hot(target, num_classes=self.num_classes) if self.pad_labels else torch.zeros(1, self.num_classes)
            return torch.cat((torch.reshape(sample, (1, -1)), pad), dim=1)
        raise StopIteration
    
    def _initialize_loader(self):
        self.loader = DataLoader(self.dataset, sampler=RandomSampler(self.dataset))
        self.current = 0


class SequentialLoader:
    """A loader class to load sample from dataset randomly
        Values:
            dataset: A dataset object
            num_classes: Number of unique classes in the dataset
    """
    def __init__(self, dataset: Dataset, num_classes: int):
        self.dataset = dataset
        self.num_classes = num_classes
        
    def __call__(self, nsamples: int, pad_label: bool=True):
        self.nsamples = nsamples
        self.pad_labels = pad_label
        self._initialize_batch()
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.nsamples:
            self.current += 1
            return torch.unsqueeze(self.batch[self.current-1,:], 0)
        raise StopIteration
    
    def _initialize_batch(self):
        """Function to initialize an ordered batch
        """
        self.batch = self._ordered_batch()
        self.current = 0
        
    def _ordered_batch(self):
        """Generates a random batch and orders it
        Returns:
            A 2D tensor of ordered batch with samples as rows. The rows can be optionally padded with one-hot label vectors 
        """
        batch, targets = next(iter(DataLoader(self.dataset, sampler=RandomSampler(self.dataset), batch_size=self.nsamples)))
        sequence = self._sequence(targets)
        pad = F.one_hot(targets[sequence], num_classes=self.num_classes) if self.pad_labels else torch.zeros(self.nsamples, self.num_classes)
        return torch.cat((torch.reshape(batch[sequence], (self.nsamples, -1)), pad), dim=1)
    
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