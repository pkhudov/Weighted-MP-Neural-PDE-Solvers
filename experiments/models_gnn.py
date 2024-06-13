import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from torch_cluster import radius_graph
from torch_geometric.data import Data
from equations.PDEs import *
from torch_geometric.nn import MessagePassing, global_mean_pool, InstanceNorm, avg_pool_x, BatchNorm

class Swish(nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)


class GNN_Layer(MessagePassing):
    """
    Message passing layer
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 time_window: int,
                 n_variables: int,
                 sigma: float,
                 smoothing: str):
        """
        Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps (temporal bundling)
            n_variables (int): number of equation specific parameters used in the solver
        """
        super(GNN_Layer, self).__init__(node_dim=-2, aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.smoothing = smoothing
        self.kernel_weights_snapshot = None
        self.input_snapshot = None
        self.positions_snapshot = None
        self.pos_transformed_snapshot = None
        if self.smoothing in ['euclidean']:
            self.a = nn.Parameter(torch.tensor(1.))
            self.sigma = nn.Parameter(torch.tensor(sigma))
        else:
            self.a = 1
            self.sigma = sigma

        if smoothing == 'default_model':
            self.message_func = self.message_default
        elif smoothing == 'euclidean':
            self.message_func = self.message_euclidean
        elif smoothing == 'euclidean+normalised':
            self.message_func = self.message_euclidean_normalized
        elif smoothing == 'mlp+net':
            self.message_func = self.message_mlp_net
        elif smoothing == 'mlp+net_no_dist':
            self.message_func = self.message_mlp_net_no_dist
        elif smoothing == 'mlp+product':
            self.message_func = self.message_mlp_product
        elif smoothing == 'norm+kernel+net':
            self.message_func = self.message_norm_kernel_to_net
        elif smoothing == 'norm+net':
            self.message_func = self.message_norm_net
        elif smoothing == 'mlp+net_no_kernel':
            self.message_func = self.message_mlp_net_no_kernel
        elif smoothing == 'norm+mlp+product':
            self.message_func = self.message_norm_mlp_product
        elif smoothing == 'norm+mlp+net':
            self.message_func = self.message_norm_mlp_net
        elif smoothing == 'norm+mlp+net_no_dist':
            self.message_func = self.message_norm_mlp_net_no_dist
        elif smoothing == 'norm+kernel+net_no_dist':
            self.message_func = self.message_norm_kernel_to_net_no_dist
        elif smoothing == 'norm+net_no_dist':
            self.message_func = self.message_norm_net_no_dist
        elif smoothing == 'linear':
            self.message = self.message_linear

        if self.smoothing in ['norm+mlp+net', 'mlp+net', 'norm+kernel+net', 'norm+net', 'mlp+net_no_kernel']:
            self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + time_window + 1 + 1 + n_variables, hidden_features),
                                           Swish()
                                           )
        else:
            self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + time_window + 1 + n_variables, hidden_features),
                                           Swish()
                                           )
            
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                           Swish()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features + n_variables, hidden_features),
                                          Swish()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features),
                                          Swish()
                                          )
        if self.smoothing in ['mlp+net', 'mlp+net_no_dist', 'mlp+product', 'mlp+net_no_kernel']:
            self.kernel_net = nn.Sequential(nn.Linear(2, hidden_features),
                                            Swish(),
                                            nn.Linear(hidden_features, 1),
                                            nn.Sigmoid()
                                            )
        elif self.smoothing in ['norm+mlp+net_no_dist', 'norm+mlp+net', 'norm+mlp+product']:
            self.kernel_net = nn.Sequential(nn.Linear(1, hidden_features),
                                            Swish(),
                                            nn.Linear(hidden_features, 1),
                                            Swish()
                                            )

        #    save_path = 'models/init5202158.pt'
        #    checkpoint = torch.load(save_path, map_location='cpu')
            
        #    self.kernel_net[0].weight.data = checkpoint['gnn_layers.3.kernel_net.0.weight']
        #    self.kernel_net[0].bias.data = checkpoint['gnn_layers.3.kernel_net.0.bias']
        #    self.kernel_net[2].weight.data = checkpoint['gnn_layers.3.kernel_net.2.weight']
        #    self.kernel_net[2].bias.data = checkpoint['gnn_layers.3.kernel_net.2.bias']

        #init.kaiming_normal_(self.kernel_net[0].weight)
        #init.constant_(self.kernel_net[0].bias, 0)

        # for  layer in self.kernel_net:
        #     if isinstance(layer, nn.Linear):
        #         if layer.out_features == 1:  # Final layer
        #             init.xavier_normal_(layer.weight) # Xavier (Sigmoid)
        #         else:
        #             init.kaiming_normal_(layer.weight) # He (ReLU)
        #         init.constant_(layer.bias, 0)

        #for  layer in self.kernel_net:
        #    if isinstance(layer, nn.Linear):
        #        if layer.out_features == 1:  # Final layer
        #            init.kaiming_normal_(layer.weight) # He (LeakyReLU)
        #        else:
        #            init.kaiming_normal_(layer.weight, nonlinearity='relu') # He (ReLU)
        #        init.constant_(layer.bias, 0)
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, variables, edge_index, batch):
        """
        Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos, variables=variables)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        """
        Message update following formula 8 of the paper
        """
        return self.message_func(x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i)

    def update(self, message, x, variables):
        """
        Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)
        if self.in_features == self.out_features:
            return x + update
        else:
            return update
    
    @staticmethod
    def gaussian_function(x, sigma, a=1.0):
        return a*torch.exp(-(x**2./sigma))
    
    def message_default(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        return message

    def message_linear(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        kernel_weights = 1/torch.norm(pos_i-pos_j, p=2., dim=-1)
        self.kernel_weights_snapshot = kernel_weights
        message = message*kernel_weights
        return message

    def message_euclidean(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        self.positions_snapshot = torch.cat((pos_i, pos_j), dim=-1)
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        kernel_weights = GNN_Layer.gaussian_function(torch.norm(pos_i - pos_j,p=2.,dim=-1), sigma=self.sigma, a=self.a).unsqueeze(-1)
        self.kernel_weights_snapshot = kernel_weights
        message = message*kernel_weights
        return message
    
    def message_euclidean_normalized(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        self.positions_snapshot = torch.cat((pos_i, pos_j), dim=-1)
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        kernel_weights = GNN_Layer.gaussian_function(torch.norm(pos_i - pos_j,p=2.,dim=-1), sigma=self.sigma).unsqueeze(-1)
        kernel_weights = kernel_weights/torch.sum(kernel_weights)
        self.kernel_weights_snapshot = kernel_weights
        message = message*kernel_weights
        return message
    
    def message_mlp_net(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        self.positions_snapshot = torch.cat((pos_i, pos_j), dim=-1)
        positions_transformed = self.kernel_net(torch.cat((pos_i, pos_j), dim=-1))
        self.pos_transformed_snapshot = positions_transformed
        kernel_weights = GNN_Layer.gaussian_function(positions_transformed, sigma=self.sigma)
        self.kernel_weights_snapshot = kernel_weights
        variables_i.requires_grad = True # To make input have grads because at least one has grad
        input = torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, kernel_weights, variables_i), dim=-1)
        if self.training:
            input.retain_grad()
        self.input_snapshot = input 
        message = self.message_net_1(input)
        message = self.message_net_2(message)
        return message
   
    def message_norm_mlp_net(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        self.positions_snapshot = torch.cat((pos_i, pos_j), dim=-1)
        norm = torch.norm(pos_i - pos_j,p=2.,dim=-1).unsqueeze(-1)
        positions_transformed = self.kernel_net(norm)
        self.pos_transformed_snapshot = positions_transformed
        kernel_weights = GNN_Layer.gaussian_function(positions_transformed, sigma=self.sigma)
        self.kernel_weights_snapshot = kernel_weights
        #variables_i.requires_grad = True # To make input have grads because at least one has grad
        input = torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, kernel_weights, variables_i), dim=-1)
        #if self.training:
        #    input.retain_grad()
        self.input_snapshot = input
        message = self.message_net_1(input)
        message = self.message_net_2(message)
        return message
 
    def message_mlp_net_no_dist(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        self.positions_snapshot = torch.cat((pos_i, pos_j), dim=-1)
        positions_transformed = self.kernel_net(torch.cat((pos_i, pos_j), dim=-1))
        self.pos_transformed_snapshot = positions_transformed
        kernel_weights = GNN_Layer.gaussian_function(positions_transformed, sigma=self.sigma)
        self.kernel_weights_snapshot = kernel_weights
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, kernel_weights, variables_i), dim=-1))
        message = self.message_net_2(message)
        return message

    def message_norm_mlp_net_no_dist(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        self.positions_snapshot = torch.cat((pos_i, pos_j), dim=-1)
        norm = torch.norm(pos_i - pos_j,p=2.,dim=-1).unsqueeze(-1)
        positions_transformed = self.kernel_net(norm)
        self.pos_transformed_snapshot = positions_transformed
        kernel_weights = GNN_Layer.gaussian_function(positions_transformed, sigma=self.sigma)
        self.kernel_weights_snapshot = kernel_weights
        #variables_i.requires_grad = True
        input = torch.cat((x_i, x_j, u_i - u_j, kernel_weights, variables_i), dim=-1)
        #if self.training:
        #    input.retain_grad()
        self.input_snapshot = input
        message = self.message_net_1(input)
        message = self.message_net_2(message)
        return message
   
    def message_mlp_product(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        self.positions_snapshot = torch.cat((pos_i, pos_j), dim=-1)
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        positions_transformed = self.kernel_net(torch.cat((pos_i, pos_j), dim=-1))
        self.pos_transformed_snapshot = positions_transformed
        kernel_weights = GNN_Layer.gaussian_function(positions_transformed, sigma=self.sigma)
        self.kernel_weights_snapshot = kernel_weights
        message = message*kernel_weights
        return message
    
    def message_norm_mlp_product(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        self.positions_snapshot = torch.cat((pos_i, pos_j), dim=-1)
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        norm = torch.norm(pos_i - pos_j,p=2.,dim=-1).unsqueeze(-1)
        positions_transformed = self.kernel_net(norm)
        self.pos_transformed_snapshot = positions_transformed
        kernel_weights = GNN_Layer.gaussian_function(positions_transformed, sigma=self.sigma)
        self.kernel_weights_snapshot = kernel_weights
        message = message*kernel_weights
        return message
    
    def message_norm_kernel_to_net(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        self.positions_snapshot = torch.cat((pos_i, pos_j), dim=-1)
        kernel_weights = GNN_Layer.gaussian_function(torch.norm(pos_i - pos_j,p=2.,dim=-1), sigma=self.sigma).unsqueeze(-1)
        self.kernel_weights_snapshot = kernel_weight
        variables_i.requires_grad = True # To make input have grads because at least one has grads
        input = torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, kernel_weights, variables_i), dim=-1)
        if self.training:
            input.retain_grad()
        self.input_snapshot = input
        message = self.message_net_1(input)
        message = self.message_net_2(message)
        return message

    def message_norm_kernel_to_net_no_dist(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        self.positions_snapshot = torch.cat((pos_i, pos_j), dim=-1)
        kernel_weights = GNN_Layer.gaussian_function(torch.norm(pos_i - pos_j,p=2.,dim=-1), sigma=self.sigma).unsqueeze(-1)
        self.kernel_weights_snapshot = kernel_weights
        variables_i.requires_grad = True
        input = torch.cat((x_i, x_j, u_i - u_j, kernel_weights, variables_i), dim=-1)
        self.input_snapshot = input
        message = self.message_net_1(input)
        message = self.message_net_2(message)
        return message
    
    def message_norm_net(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        self.positions_snapshot = torch.cat((pos_i, pos_j), dim=-1)
        norm = torch.norm(pos_i - pos_j,p=2.,dim=-1).unsqueeze(-1)
        #norm.requires_grad = True # To make input have grads because at least one has grad
        input = torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, norm, variables_i), dim=-1)
        #if self.training:
        #    input.retain_grad()
        self.input_snapshot = input
        message = self.message_net_1(input)
        message = self.message_net_2(message)
        return message

    def message_norm_net_no_dist(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        self.positions_snapshot = torch.cat((pos_i, pos_j), dim=-1)
        norm = torch.norm(pos_i - pos_j,p=2.,dim=-1).unsqueeze(-1)
        input = torch.cat((x_i, x_j, u_i - u_j, norm, variables_i), dim=-1)
        message = self.message_net_1(input)
        message = self.message_net_2(message)
        return message
    
    def message_mlp_net_no_kernel(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        self.positions_snapshot = torch.cat((pos_i, pos_j), dim=-1)
        positions_transformed = self.kernel_net(torch.cat((pos_i, pos_j), dim=-1))
        self.pos_transformed_snapshot = positions_transformed
        variables_i.requires_grad = True # To make input have grads because at least one has grad
        input = torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, positions_transformed, variables_i), dim=-1)
        if self.training:
            input.retain_grad()
        self.input_snapshot = input
        message = self.message_net_1(input)
        message = self.message_net_2(message)
        return message


class MP_PDE_Solver(torch.nn.Module):
    """
    MP-PDE solver class
    """
    def __init__(self,
                 pde: PDE,
                 time_window: int = 25,
                 hidden_features: int = 164,
                 hidden_layer: int = 6,
                 eq_variables: dict = {},
                 sigma: float = 0.1,
                 smoothing: str = 'euclidean'
    ):
        """
        Initialize MP-PDE solver class.
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            pde (PDE): PDE at hand
            time_window (int): number of input/output timesteps (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
            eq_variables (dict): dictionary of equation specific parameters
        """
        super(MP_PDE_Solver, self).__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        assert(time_window == 25 or time_window == 20 or time_window == 50)
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables
        self.sigma = sigma
        self.smoothing = smoothing

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            n_variables=len(self.eq_variables) + 1,  # variables = eq_variables + time
            sigma=self.sigma,
            smoothing=self.smoothing
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.gnn_layers.append(GNN_Layer(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=len(self.eq_variables) + 1,
                                         sigma=self.sigma,
                                         smoothing=self.smoothing
                                        )
                               )

        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.time_window + 2 + len(self.eq_variables), self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
        )


        # Decoder CNN, maps to different outputs (temporal bundling)
        if(self.time_window==20):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 15, stride=4),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )
        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 16, stride=4),
                                            Swish(),
                                            nn.Conv1d(8, 1, 14, stride=1)
                                            )
        if(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )

    def __repr__(self):
        return f'GNN'

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
        Returns:
            torch.Tensor: data output
        """
        u = data.x
        # Encode and normalize coordinate information
        pos = data.pos
        pos_x = pos[:, 1][:, None] / self.pde.L
        pos_t = pos[:, 0][:, None] / self.pde.tmax
        edge_index = data.edge_index
        batch = data.batch

        # Encode equation specific parameters
        # alpha, beta, gamma are used in E1,E2,E3 experiments
        # bc_left, bc_right, c are used in WE1, WE2, WE3 experiments
        variables = pos_t    # time is treated as equation variable
        if "alpha" in self.eq_variables.keys():
            variables = torch.cat((variables, data.alpha / self.eq_variables["alpha"]), -1)
        if "beta" in self.eq_variables.keys():
            variables = torch.cat((variables, data.beta / self.eq_variables["beta"]), -1)
        if "gamma" in self.eq_variables.keys():
            variables = torch.cat((variables, data.gamma / self.eq_variables["gamma"]), -1)
        if "bc_left" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_left), -1)
        if "bc_right" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_right), -1)
        if "c" in self.eq_variables.keys():
            variables = torch.cat((variables, data.c / self.eq_variables["c"]), -1)

        # Encoder and processor (message passing)
        node_input = torch.cat((u, pos_x, variables), -1)
        h = self.embedding_mlp(node_input)
        for i in range(self.hidden_layer):
            h = self.gnn_layers[i](h, u, pos_x, variables, edge_index, batch)

        # Decoder (formula 10 in the paper)
        dt = (torch.ones(1, self.time_window) * self.pde.dt).to(h.device)
        dt = torch.cumsum(dt, dim=1)
        # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
        diff = self.output_mlp(h[:, None]).squeeze(1)
        out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff

        return out
