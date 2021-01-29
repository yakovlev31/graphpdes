import torch.nn as nn


class DynamicsFunction(nn.Module):
    def __init__(self, model, params=None):
        super().__init__()
        self.model = model
        if params is None:
            self.params = {}
        else:
            self.params = params
    
    def forward(self, t, u):
        return self.model(u, **self.params)

    def update_params(self, params):
        self.params.update(params)
