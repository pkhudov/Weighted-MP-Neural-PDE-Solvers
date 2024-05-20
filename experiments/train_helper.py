import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader
from common.utils import HDF5Dataset, GraphCreator
from equations.PDEs import *
from typing import Tuple
from torch_geometric.data import Data


def print_kernel_weights(model, graph):
    kernel_weights = torch.stack([layer.kernel_weights_snapshot for layer in model.gnn_layers])
    
    positions = model.gnn_layers[0].positions_snapshot
    neighbours_left = graph.edge_index[0, 51:57]
    neighbours_right = graph.edge_index[0, 57:63]
    positions_left = positions[51:57, 1]
    positions_right = positions[57:63, 1]
    kernel_weights_left = kernel_weights[:, 51:57]
    kernel_weights_right = kernel_weights[:, 57:63]

    if model.gnn_layers[0].pos_transformed_snapshot is not None:
        pos_transformed = torch.stack([layer.pos_transformed_snapshot for layer in model.gnn_layers])
    else:
        pos_transformed = None

    print('='*105)
    if pos_transformed is not None:
        transformed_left = pos_transformed[:, 51:57]
        transformed_right = pos_transformed[:, 57:63]
        
        print('Positions transformed for neighbours of Node 6 (for each GNN layer):\n')
        for i in range(6):
            pos_tr = [f'{transformed_left[l, i].item():.5e}' for l in range(6)] # 6 is the number of layers
            pos_tr = ', '.join(pos_tr)
            print(f'Node: {neighbours_left[i]:02d} (pos: {positions_left[i]:.3f}): {pos_tr}')
        print ('-')
        for i in range(6):
            pos_tr = [f'{transformed_right[l, i].item():.5e}' for l in range(6)]
            pos_tr = ', '.join(pos_tr)
            print(f'Node: {neighbours_right[i]:02d} (pos: {positions_right[i]:.3f}): {pos_tr}')

    print('\nKernel Weights for neighbours of Node 6 (for each GNN layer):\n')
    for i in range(6):
        weights = [f'{kernel_weights_left[l, i].item():.5e}' for l in range(6)] # 6 is the number of layers
        weights = ', '.join(weights)
        print(f'Node: {neighbours_left[i]:02d} (pos: {positions_left[i]:.3f}): {weights}')
    print ('-')
    for i in range(6):
        weights = [f'{kernel_weights_right[l, i].item():.5e}' for l in range(6)]
        weights = ', '.join(weights)
        print(f'Node: {neighbours_right[i]:02d} (pos: {positions_right[i]:.3f}): {weights}')
    print('-'*105)
    
    # For some reason the adjacency list lost its order, so left and right is not correct here
    neighbours_left1 = graph.edge_index[0, 651:657]
    neighbours_right1 = graph.edge_index[0, 657:663]
    positions_left1 = positions[651:657, 1]
    positions_right1 = positions[657:663, 1]
    kernel_weights_left1 = kernel_weights[:, 651:657]
    kernel_weights_right1 = kernel_weights[:, 657:663]

    if pos_transformed is not None:
        transformed_left = pos_transformed[:, 651:657]
        transformed_right = pos_transformed[:, 657:663]
        print('Positions transformed for neighbours of Node 56 (for each GNN layer):\n')
        for i in range(6):
            pos_tr = [f'{transformed_left[l, i].item():.5e}' for l in range(6)] # 6 is the number of layers
            pos_tr = ', '.join(pos_tr)
            print(f'Node: {neighbours_left[i]:02d} (pos: {positions_left[i]:.3f}): {pos_tr}')
        print ('-')
        for i in range(6):
            pos_tr = [f'{transformed_right[l, i].item():.5e}' for l in range(6)]
            pos_tr = ', '.join(pos_tr)
            print(f'Node: {neighbours_right[i]:02d} (pos: {positions_right[i]:.3f}): {pos_tr}')

    print('\nKernel Weights for neighbours of Node 56 (for each GNN layer):\n')
    for i in range(6):
        weights = [f'{kernel_weights_left1[l, i].item():.5e}' for l in range(6)] # 6 is the number of layers
        weights = ', '.join(weights)
        print(f'Node: {neighbours_left1[i]:02d} (pos: {positions_left1[i]:.3f}): {weights}')
    print ('-')
    for i in range(6):
        weights = [f'{kernel_weights_right1[l, i].item():.5e}' for l in range(6)]
        weights = ', '.join(weights)
        print(f'Node: {neighbours_right1[i]:02d} (pos: {positions_right1[i]:.3f}): {weights}')
    print('='*105)

def print_kernel_net_gradients(model):
    for l, layer in enumerate(model.gnn_layers):
        gradients = []
        for name, parameter in layer.kernel_net.named_parameters():
            if parameter.grad is not None:
                gradients.append((name, parameter.grad.norm().item()))
            else:
                gradients.append((name, None))
        print(f'GNN_Layer {l} gradients:')
        print(gradients, '\n')


def training_loop(model: torch.nn.Module,
                  unrolling: list,
                  batch_size: int,
                  optimizer: torch.optim,
                  loader: DataLoader,
                  graph_creator: GraphCreator,
                  criterion: torch.nn.modules.loss,
                  i: int,
                  epoch: int,
                  device: torch.cuda.device="cpu") -> Tuple[torch.Tensor, Data]:
    """
    One training epoch with random starting points for every trajectory
    Args:
        model (torch.nn.Module): neural network PDE solver
        unrolling (list): list of different unrolling steps for each batch entry
        batch_size (int): batch size
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        (torch.Tensor, Data): training losses and graph data
    """

    losses = []
    for j, (u_base, u_super, x, variables) in enumerate(loader):
        optimizer.zero_grad()
        # Randomly choose number of unrollings
        unrolled_graphs = random.choice(unrolling)
        steps = [t for t in range(graph_creator.tw,
                                  graph_creator.t_res - graph_creator.tw - (graph_creator.tw * unrolled_graphs) + 1)]
        # Randomly choose starting (time) point at the PDE solution manifold
        random_steps = random.choices(steps, k=batch_size)
        data, labels = graph_creator.create_data(u_super, random_steps)
        if f'{model}' == 'GNN':
            graph = graph_creator.create_graph(data, labels, x, variables, random_steps).to(device)
        else:
            data, labels = data.to(device), labels.to(device)

        # Unrolling of the equation which serves as input at the current step
        # This is the pushforward trick!!!
        with torch.no_grad():
            for _ in range(unrolled_graphs):
                random_steps = [rs + graph_creator.tw for rs in random_steps]
                _, labels = graph_creator.create_data(u_super, random_steps)
                if f'{model}' == 'GNN':
                    pred = model(graph)
                    graph = graph_creator.create_next_graph(graph, pred, labels, random_steps).to(device)
                else:
                    data = model(data)
                    labels = labels.to(device)

        if f'{model}' == 'GNN':
            pred = model(graph)
            loss = criterion(pred, graph.y)
        else:
            pred = model(data)
            loss = criterion(pred, labels)

        loss = torch.sqrt(loss)
        loss.backward()
        losses.append(loss.detach() / batch_size)
        optimizer.step()

        # Print kernel weights for the first forward pass of the epoch
        if epoch == 0 and i == 0 and j == 0 and model.gnn_layers[0].kernel_weights_snapshot is not None:
            print_kernel_weights(model, graph)

    losses = torch.stack(losses)
    return losses, graph

def test_timestep_losses(model: torch.nn.Module,
                         steps: list,
                         batch_size: int,
                         loader: DataLoader,
                         graph_creator: GraphCreator,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu") -> None:
    """
    Loss for one neural network forward pass at certain timepoints on the validation/test datasets
    Args:
        model (torch.nn.Module): neural network PDE solver
        steps (list): input list of possible starting (time) points
        batch_size (int): batch size
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """

    for step in steps:

        if (step != graph_creator.tw and step % graph_creator.tw != 0):
            continue

        losses = []
        for (u_base, u_super, x, variables) in loader:
            with torch.no_grad():
                same_steps = [step]*batch_size
                data, labels = graph_creator.create_data(u_super, same_steps)
                if f'{model}' == 'GNN':
                    graph = graph_creator.create_graph(data, labels, x, variables, same_steps).to(device)
                    pred = model(graph)
                    loss = criterion(pred, graph.y)
                else:
                    data, labels = data.to(device), labels.to(device)
                    pred = model(data)
                    loss = criterion(pred, labels)
                losses.append(loss / batch_size)

        losses = torch.stack(losses)
        print(f'Step {step}, mean loss {torch.mean(losses)}')


def test_unrolled_losses(model: torch.nn.Module,
                         steps: list,
                         batch_size: int,
                         nr_gt_steps: int,
                         nx_base_resolution: int,
                         loader: DataLoader,
                         graph_creator: GraphCreator,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu") -> torch.Tensor:
    """
    Loss for full trajectory unrolling, we report this loss in the paper
    Args:
        model (torch.nn.Module): neural network PDE solver
        steps (list): input list of possible starting (time) points
        nr_gt_steps (int): number of numerical input timesteps
        nx_base_resolution (int): spatial resolution of numerical baseline
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: valid/test losses
    """
    losses = []
    losses_base = []
    for (u_base, u_super, x, variables) in loader:
        losses_tmp = []
        losses_base_tmp = []
        with torch.no_grad():
            same_steps = [graph_creator.tw * nr_gt_steps] * batch_size
            data, labels = graph_creator.create_data(u_super, same_steps)
            if f'{model}' == 'GNN':
                graph = graph_creator.create_graph(data, labels, x, variables, same_steps).to(device)
                pred = model(graph)
                loss = criterion(pred, graph.y) / nx_base_resolution
            else:
                data, labels = data.to(device), labels.to(device)
                pred = model(data)
                loss = criterion(pred, labels) / nx_base_resolution

            losses_tmp.append(loss / batch_size)

            # Unroll trajectory and add losses which are obtained for each unrolling
            for step in range(graph_creator.tw * (nr_gt_steps + 1), graph_creator.t_res - graph_creator.tw + 1, graph_creator.tw):
                same_steps = [step] * batch_size
                _, labels = graph_creator.create_data(u_super, same_steps)
                if f'{model}' == 'GNN':
                    graph = graph_creator.create_next_graph(graph, pred, labels, same_steps).to(device)
                    pred = model(graph)
                    loss = criterion(pred, graph.y) / nx_base_resolution
                else:
                    labels = labels.to(device)
                    pred = model(pred)
                    loss = criterion(pred, labels) / nx_base_resolution
                losses_tmp.append(loss / batch_size)

            # Losses for numerical baseline
            for step in range(graph_creator.tw * nr_gt_steps, graph_creator.t_res - graph_creator.tw + 1,
                              graph_creator.tw):
                same_steps = [step] * batch_size
                _, labels_super = graph_creator.create_data(u_super, same_steps)
                _, labels_base = graph_creator.create_data(u_base, same_steps)
                loss_base = criterion(labels_super, labels_base) / nx_base_resolution
                losses_base_tmp.append(loss_base / batch_size)

        losses.append(torch.sum(torch.stack(losses_tmp)))
        losses_base.append(torch.sum(torch.stack(losses_base_tmp)))

    losses = torch.stack(losses)
    losses_base = torch.stack(losses_base)
    print(f'Unrolled forward losses {torch.mean(losses)}')
    print(f'Unrolled forward base losses {torch.mean(losses_base)}')

    return losses




