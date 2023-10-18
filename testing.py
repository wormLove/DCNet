import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformations import Transform
from tqdm import tqdm


class GenerateEmbeddings:
    def __call__(self, layer: nn.Module, input: torch.Tensor):
        layer.test_mode('on')
        out = layer(input)
        layer.test_mode('off')
        return out

class TestModule:
    def __init__(self, test_metric: str = 'generate_outputs'):
        self.test_metric = GenerateEmbeddings()
        
    def __call__(self, network: nn.Module, test_dataset: Dataset, input_transforms: Transform, nsamples: int = None):
        self.network = network
        self.test_dataset = test_dataset
        self.input_transforms = input_transforms
        self.nsamples = nsamples if nsamples is not None else len(test_dataset)
        self.output = {idx: torch.empty(0) for idx, _ in enumerate(network)}
        self.targets = []
        
        self._metric_call()
        return self.output, self.targets
        
    def _metric_call(self):
        loader = DataLoader(self.test_dataset)
        for inp_idx, (input, target) in enumerate(tqdm(loader, total=len(loader), desc='Testing')):
            if inp_idx < self.nsamples:
                self.targets.append(target.item())
                out_ = self.input_transforms(input, label=target.item())
                for idx, layer in enumerate(self.network):
                    out_ = self.test_metric(layer, out_)
                    self.output[idx] = torch.cat((self.output[idx], out_), dim=0)
