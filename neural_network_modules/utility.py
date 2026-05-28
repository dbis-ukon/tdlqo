import logging
from typing import List, Tuple

import torch
import prettytable


def build_layer_stack(previous: int,
                      layers: List[int],
                      activation: torch.nn.Module,
                      dropout: float = 0
                      ) -> Tuple[torch.nn.ModuleList, int]:
    modules = torch.nn.ModuleList()
    for layer in layers:
        modules.append(torch.nn.Linear(previous, layer))
        if isinstance(activation, torch.nn.PReLU):
            modules.append(torch.nn.PReLU(num_parameters=layer))
        else:
            modules.append(activation)
        if dropout > 0:
            modules.append(torch.nn.Dropout(dropout))
        previous = layer
    return modules, previous


def print_parameters(module: torch.nn.Module, logger: logging.Logger):
    # Print the parameters of a PyTorch module in a table format
    table = prettytable.PrettyTable()
    table.field_names = ["Parameter Name", "Number of Parameters"]
    total_params = 0
    for name, param in module.named_parameters():
        num_params = param.numel()
        total_params += num_params
        table.add_row([name, num_params])
    table.add_row(["Total", total_params])
    logger.info("\n" + table.get_string())


def check_model_corruption(module: torch.nn.Module) -> bool:
    for param in module.parameters():
        if not torch.isfinite(param).all():
            return True
    return False
