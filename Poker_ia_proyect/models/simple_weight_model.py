import torch
import torch.nn as nn

class SimpleWeightModel(nn.Module):
    """
    Modelo simple para aprender los pesos de las estrategias.
    """
    def __init__(self, num_strategies: int, state_size: int):
        super(SimpleWeightModel, self).__init__()
        self.linear = nn.Linear(state_size, num_strategies)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.linear(state)
        x = self.softmax(x)
        return x