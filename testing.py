import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformations import Transform


class BaseTester:
    def __call__(self, network: nn.Module, input: torch.Tensor):
        out = torch.empty(0)
        for layer in network:
            layer.test_mode('on')
            out = torch.cat((out, layer(input)), dim=0)
            layer.test_mode('off')
        return out


class ModuleTester:
    def __init__(self, dataset: Dataset, transforms: Transform):
        self.dataset = dataset
        self.transforms = transforms
        self.loader = DataLoader(dataset, shuffle=True)
        self.tester = BaseTester()
    
    def __call__(self, network: nn.Module, nsamples: int):
        self.network = network
        self.nsamples = nsamples
        self.current = 0
        return self
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.nsamples:
            input, target = next(iter(self.loader))
            input = self.transforms(input, label=target.item())
            out = self.tester(self.network, input)
            self.current += 1
            return out, target.item()
        raise StopIteration
        