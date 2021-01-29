# Continuous-time PDE models on unstructured grids

Implementation of models from [Learning continuous-time PDEs from sparse data with graph neural networks](https://openreview.net/forum?id=aUX5Plaq7Oy).


This package is based on [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric).

Installation: python -m pip install git+https://github.com/yakovlev31/graphpdes.git

Basic usage example:

```python
# Setup grid for pytorch_geometric
edge_index = torch.tensor(
    [
        [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4], 
        [1, 4, 0, 3, 3, 4, 1, 2, 4, 3, 0, 2]
    ], 
    dtype=torch.long,
)
rel_pos = torch.tensor(
    [
        [1.0, 0.0], [0.0, 1.0], 
        [-1.0, 0.0], [0.0, 1.0], 
        [0.707106781, 0.707106781], 
        [-0.707106781, 0.707106781], 
        [0.0, -1.0], [-0.707106781, -0.707106781], 
        [-1.0, 0.0], [1.0, 0.0], 
        [0.0, -1.0], 
        [0.707106781, -0.707106781],
    ], 
    dtype=torch.float
)
u = torch.tensor([
    [0.0, 0.0], [1.0, 1.0], 
    [2.0, 2.0], [3.0, 3.0], 
    [4.0, 4.0]], dtype=torch.float)

# Create model and dynamics function
gamma = nn.Linear(3, 2)
phi = nn.Linear(6, 1)
model = Model(gamma, phi)
F = DynamicsFunction(model)

# Create model and dynamics function (with Dirichlet boundary conditions)
gamma_dir = nn.Linear(3, 2)
phi_dir = nn.Linear(6, 1)
model_dir = ModelDirichlet(gamma_dir, phi_dir)
F_dir = DynamicsFunction(model_dir)
bcs_dict = {"bc_0": [[0, 3, 2], [0]], "bc_1": [[1, 4], [1]]}  # {bc_name: [[node_inds], [field_inds]], etc.}

# Calculate dudt for given grid and u
params_dict = {'edge_index': edge_index, 'rel_pos': rel_pos}
F.update_params(params_dict)
dudt = F(u)

# Calculate dudt for given grid and u (with Dirichlet boundary conditions)
params_dict_dir = {'edge_index': edge_index, 'rel_pos': rel_pos, 'bcs_dict': bcs_dict}
F_dir.update_params(params_dict_dir)
dudt_dir = F_dir(u)
```

The time derivatives could then be plugged into and ODE solver to simulate the dynamics. Examples of using this package with [torchdiffeq](https://github.com/rtqichen/torchdiffeq) could be found in [graphpdes_experiments](https://github.com/yakovlev31/graphpdes_experiments/).
