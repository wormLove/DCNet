import torch
from torchmetrics import Metric
from collections import defaultdict

class Conncetedness(Metric):
    """A metric class to measure the self connectedness of a discrimination module
    """
    def __init__(self):
        super().__init__()
        self.total = defaultdict(list)
        self.correct = defaultdict(list)
        self.wrong = defaultdict(list)
    
    def update(self, graph: torch.Tensor, labels: torch.Tensor):
        assert graph.dim() == 2 and labels.dim() == 1
        assert graph.shape[0] == graph.shape[1] == len(labels)
        
        for idx, row in enumerate(graph):
            row_label = labels[idx]
            connected_labels = labels[row.nonzero().squeeze()]
            disconnected_labels =  labels[(row == 0).nonzero().squeeze()]
            self.correct[row_label.item()].append(torch.sum(row_label == connected_labels).item())
            self.wrong[row_label.item()].append(torch.sum(row_label == disconnected_labels).item())
            self.total[row_label.item()].append(connected_labels.numel())
    
    def compute(self):
        N = len(self.correct)
        precision = 0
        recall = 0
        for label, values in self.correct.items():
            idx = torch.argmax(torch.tensor(values))
            correct = values[idx.item()]
            wrong = self.wrong[label][idx.item()]
            total = self.total[label][idx.item()]
            
            precision += correct/total
            recall += correct/(correct + wrong)
        
        return precision/N, recall/N
    
    def reset(self):
        self.correct = defaultdict(list)
        self.wrong = defaultdict(list)
        self.total = defaultdict(list)

class Consistency(Metric):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.consistency = 0.0
        
    def update(self, labels: torch.Tensor):
        assert labels.dim() == 1
        if self.labels is not None:
            self.consistency = torch.sum(self.labels == labels).item()/len(labels)
        self.labels = labels
        
    def compute(self):
        return self.consistency
    
    def reset(self):
        self.labels = None
        self.consistency = 0.0
        
class Degree(Metric):
    def __init__(self):
        super().__init__()
        self.degree = 0
        
    def update(self, weights: torch.Tensor):
        self.degree = torch.mean(torch.sum(weights, dim=0)).item()
    
    def compute(self):
        return self.degree
    
    def reset(self):
        pass
