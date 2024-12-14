# class DeepMAgeModel(nn.Module):
#     """Deep neural network for age prediction."""
#
#     def __init__(self, config):
#         super(DeepMAgeModel, self).__init__()
#         self.config = config
#
#         hidden_edges = self.config["model.hidden_edges"]
#         if isinstance(hidden_edges, str):
#             hidden_edges = json.loads(hidden_edges)
#
#         # Dynamically create layers.
#         self.fcs = nn.ModuleList()
#         self.fcs.append(nn.Linear(self.config["model.input_dim"], hidden_edges[0]))
#         previous_out = hidden_edges[0]
#         for inner_layer in hidden_edges[1:]:
#             self.fcs.append(nn.Linear(previous_out, inner_layer))
#             previous_out = inner_layer
#         self.fcs.append(nn.Linear(previous_out, 1))
#
#         self.dropout = nn.Dropout(self.config["model.dropout"])
#         activation_funcs = {
#             "celu": nn.CELU(),
#             "elu": nn.ELU(),
#             "gelu": nn.GELU(),
#             "leakyrelu": nn.LeakyReLU(),
#             "relu": nn.ReLU(),
#             "silu": nn.SiLU(),
#         }
#         self.activation_func = activation_funcs[self.config["model.activation_func"]]
#
#     def forward(self, x):
#         for i, fc in enumerate(self.fcs[:-1]):
#             x = self.activation_func(fc(x))
#             # Add dropout only if not the final hidden layer
#             if i < len(self.fcs[:-1]) - 1:
#                 x = self.dropout(x)
#         x = self.fcs[-1](x)
#         return x

# class DeepMAgeModel(nn.Module):
#     """Deep neural network for age prediction with configurable inner layers."""
#
#     def __init__(self, config):
#         super(DeepMAgeModel, self).__init__()
#         self.config = config
#         hidden_edges = self.config["hidden_edges"]
#         dropout = self.config["dropout"]
#         activation_funcs = {
#             "elu": nn.ELU(),
#             "relu": nn.ReLU(),
#         }
#         activation_func = activation_funcs[self.config["activation_func"]]
#
#         # Create the layers dynamically
#         layers = []
#         input_dim = self.config["input_dim"]
#         for i, layer_dim in enumerate(hidden_edges):
#             layers.append(nn.Linear(input_dim, layer_dim))
#             layers.append(activation_func)
#             # Add dropout only if not the final hidden layer
#             if i < len(hidden_edges) - 1:
#                 layers.append(nn.Dropout(dropout))
#             input_dim = layer_dim
#
#         # Add the final layer (output layer)
#         layers.append(nn.Linear(input_dim, 1))
#
#         # Use nn.Sequential to combine all layers into a single module
#         self.network = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.network(x)
