import torch
from torchmetrics import Metric

class Conncetedness(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='mean')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='mean')

    def update(self, graph: torch.Tensor, labels: torch.Tensor):
        assert graph.dim() == 2 and labels.dim() == 1
        assert graph.shape[0] == graph.shape[1] == len(labels)

        for idx, row in enumerate(graph):
            row_label = labels[idx]
            connected_labels = labels[row.nonzero().squeeze()]
            self.correct += torch.sum(row_label == connected_labels)
            self.total += torch.tensor(connected_labels.numel())
    
    def compute(self):
        return self.correct.item() / self.total.item() # type: ignore