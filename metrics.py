import torch
from torchmetrics import Metric

class Conncetedness(Metric):
    """A metric class to measure the self connectedness of a discrimination module
    """
    def __init__(self):
        super().__init__()
        self.total = 0
        self.correct = 0
        self.degree = 0
    
    def update(self, graph: torch.Tensor, labels: torch.Tensor):
        assert graph.dim() == 2 and labels.dim() == 1
        assert graph.shape[0] == graph.shape[1] == len(labels)
        
        graph.fill_diagonal_(0.0)
        for idx, row in enumerate(graph):
            row_label = labels[idx]
            connected_labels = labels[row.nonzero().squeeze()]
            self.correct += torch.sum(row_label == connected_labels).item()
            self.total += connected_labels.numel()
        self.degree = 0.5*self.total/len(labels)
    
    def compute(self):
        return self.correct/self.total
    
    def reset(self):
        self.correct = 0
        self.total = 0