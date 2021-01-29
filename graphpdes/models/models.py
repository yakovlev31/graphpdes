import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class Model(MessagePassing):
    def __init__(self, gamma, phi):
        super(Model, self).__init__(aggr='mean', flow='target_to_source')
        self.gamma = gamma
        self.phi = phi

    def forward(self, u, edge_index, rel_pos):
        return self.propagate(edge_index, u=u, rel_pos=rel_pos)

    def message(self, u_i, u_j, rel_pos):
        phi_input = torch.cat([u_i, u_j-u_i, rel_pos], dim=1)
        return self.phi(phi_input)

    def update(self, aggr, u):
        gamma_input = torch.cat([u, aggr], dim=1)
        dudt = self.gamma(gamma_input)
        return dudt


class ModelDirichlet(Model):
    def forward(self, u, edge_index, rel_pos, bcs_dict):
        return self.propagate(edge_index, u=u, rel_pos=rel_pos, bcs_dict=bcs_dict)
    
    def update(self, aggr, u, bcs_dict):
        dudt = super().update(aggr, u)
        for bc_inds, field_inds in bcs_dict.values():
            dudt[bc_inds, field_inds] *= 0  
        return dudt


if __name__ == "__main__":
    import torch  # noqa
    import torch.nn as nn  # noqa
    from torch_geometric.data import Data  # noqa

    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4], 
            [1, 4, 0, 3, 3, 4, 1, 2, 4, 3, 0, 2]
        ], 
        dtype=torch.long,
    )
    rel_pos = torch.tensor(
        [
            [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [0.707106781, 0.707106781], [-0.707106781, 0.707106781], [0.0, -1.0],
            [-0.707106781, -0.707106781], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.707106781, -0.707106781],
        ], 
        dtype=torch.float
    )
    u = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=torch.float)

    gamma = nn.Linear(3, 2)
    phi = nn.Linear(6, 1)

    model = Model(gamma, phi)
    dudt = model(u, edge_index, rel_pos)
    print("standard", dudt)

    bcs_dict = {"bc_0": [[0, 3, 2], [0]], "bc_1": [[1, 4], [1]]}
    model = ModelDirichlet(gamma, phi)
    dudt = model(u, edge_index, rel_pos, bcs_dict)
    print("dirichlet", dudt)
