# class DeepMAgeModel(nn.Module):
#     """Deep neural network for age prediction."""
#
#     def __init__(self, config):
#         super(DeepMAgeModel, self).__init__()
#         self.config = config
#         self.fc1 = nn.Linear(self.config["input_dim"], self.config["layer2_in"])
#         self.fc2 = nn.Linear(self.config["layer2_in"], self.config["layer3_in"])
#         self.fc3 = nn.Linear(self.config["layer3_in"], self.config["layer4_in"])
#         self.fc4 = nn.Linear(self.config["layer4_in"], self.config["layer5_in"])
#         self.fc5 = nn.Linear(self.config["layer5_in"], 1)
#         self.dropout = nn.Dropout(self.config["dropout"])
#         activation_funcs = {
#             "elu": nn.ELU(),
#             "relu": nn.ReLU(),
#         }
#         self.activation_func = activation_funcs[self.config["activation_func"]]
#
#     def forward(self, x):
#         x = self.activation_func(self.fc1(x))
#         x = self.dropout(x)
#         x = self.activation_func(self.fc2(x))
#         x = self.dropout(x)
#         x = self.activation_func(self.fc3(x))
#         x = self.dropout(x)
#         x = self.activation_func(self.fc4(x))
#         x = self.dropout(x)
#         x = self.fc5(x)
#         return x
# class DeepMAgeModel(nn.Module):
#     """Deep neural network for age prediction with configurable inner layers."""
#
#     def __init__(self, config):
#         super(DeepMAgeModel, self).__init__()
#         self.config = config
#         inner_layers = self.config["inner_layers"]
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
#         for i, layer_dim in enumerate(inner_layers):
#             layers.append(nn.Linear(input_dim, layer_dim))
#             layers.append(activation_func)
#             # Add dropout only if not the final hidden layer
#             if i < len(inner_layers) - 1:
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
