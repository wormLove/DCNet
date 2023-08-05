import torch
from torchmetrics import Metric

class Conncetedness(Metric):
    """A metric class to measure the self connectedness of a discrimination module
    """
    def __init__(self):
        super().__init__()
        self.total = 0
        self.correct = 0
        self.wrong = 0
    
    def update(self, graph: torch.Tensor, labels: torch.Tensor):
        assert graph.dim() == 2 and labels.dim() == 1
        assert graph.shape[0] == graph.shape[1] == len(labels)
        
        #graph.fill_diagonal_(0.0)
        for idx, row in enumerate(graph):
            row_label = labels[idx]
            connected_labels = labels[row.nonzero().squeeze()]
            disconnected_labels =  labels[(row == 0).nonzero().squeeze()]
            self.correct += torch.sum(row_label == connected_labels).item()
            self.wrong += torch.sum(row_label == disconnected_labels).item()
            self.total += connected_labels.numel()
        
        #self.degree = torch.sum(graph)/len(labels)
    
    def compute(self):
        return self.correct/self.total, self.correct/(self.correct + self.wrong)
    
    def reset(self):
        self.correct = 0
        self.wrong = 0
        self.total = 0
        
class Consistency(Metric):
    def __init__(self, out_dim: int):
        super().__init__()
        self.labels = torch.empty(out_dim, dtype=int)
        self.consistency = 0.0
        
    def update(self, labels: torch.Tensor):
        assert labels.dim() == 1
        
        self.consistency = torch.sum(self.labels == labels).item()/len(labels)
        self.labels = labels
        
    def compute(self):
        return self.consistency
    
    def reset(self):
        self.labels.fill_(0)
        self.consistency = 0.0
        