from typing import Tuple, Optional

import torch
import numpy as np
import torch_geometric.nn.aggr
from torch.nn import Parameter

from neural_network_modules.aggregation.multi_head_aggregation import MultiHeadAggregation
from neural_network_modules.message_passing.multi_head_attention_convolution_index import MultiHeadAttentionConvolutionIndex
from neural_network_modules.utility import build_layer_stack
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_configuration import \
    TopDownLearnedOptimizerConfiguration, ActivationFunction, TrainingQuerySelectionMode
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_data.top_down_learned_optimizer_data import TopDownLearnedOptimizerData
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_data.top_down_learned_optimizer_target_data import \
    TopDownLearnedOptimizerTargetData
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_data.top_down_learned_optimizer_training_data import \
    TopDownLearnedOptimizerTrainingData
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_encoder import TopDownLearnedOptimizerEncoder


class TopDownLearnedOptimizerModule(torch.nn.Module):
    def __init__(self,
                 configuration: TopDownLearnedOptimizerConfiguration,
                 encoder: TopDownLearnedOptimizerEncoder):
        super().__init__()
        self._configuration = configuration

        if configuration.default_activation_function == ActivationFunction.RELU:
            self._default_activation_function = torch.nn.ReLU()
        elif configuration.default_activation_function == ActivationFunction.LEAKY_RELU:
            self._default_activation_function = torch.nn.LeakyReLU()
        elif configuration.default_activation_function == ActivationFunction.PRELU:
            self._default_activation_function = torch.nn.PReLU()
        else:
            raise NotImplementedError()

        table_specific_node_size = encoder.table_specific_information_size()
        if table_specific_node_size > 0:
            self._table_specific_weights = Parameter(torch.zeros((encoder.num_tables(), configuration.table_specific_hidden_size, table_specific_node_size)))
            self._table_specific_bias = Parameter(torch.zeros((encoder.num_tables(), configuration.table_specific_hidden_size)))
            table_specific_hidden_size = configuration.table_specific_hidden_size
        else:
            self._table_specific_weights = None
            self._table_specific_bias = None
            table_specific_hidden_size = 0

        node_size = encoder.table_occurrence_information_size() + table_specific_hidden_size
        self._node_layers, node_size = build_layer_stack(node_size, configuration.node_layer_sizes, self._default_activation_function)
        edge_size = encoder.join_information_size()
        self._edge_layers, edge_size = build_layer_stack(edge_size, configuration.edge_layer_sizes, self._default_activation_function)
        self._message_passing_layers = torch.nn.ModuleList()
        node_sizes = [node_size]
        for message_passing_layer_configuration in configuration.message_passing_layer_configurations:
            self._message_passing_layers.append(MultiHeadAttentionConvolutionIndex(node_size, edge_size, message_passing_layer_configuration.output_size, message_passing_layer_configuration.message_size, message_passing_layer_configuration.num_heads, message_passing_layer_configuration.head_size, message_passing_layer_configuration.count, self._default_activation_function))
            node_size = message_passing_layer_configuration.output_size
            node_sizes.append(node_size)

        if configuration.message_passing_skip_connections:
            final_node_size = sum(node_sizes)
        else:
            final_node_size = node_size

        self._all_node_aggregation = MultiHeadAggregation(final_node_size, configuration.all_node_aggregation.output_size_per_head, configuration.all_node_aggregation.num_heads, configuration.all_node_aggregation.count)

        self._join_node_aggregation = MultiHeadAggregation(final_node_size + encoder.join_operator_subgroup_table_information_size(), configuration.join_node_aggregation.output_size_per_head, configuration.join_node_aggregation.num_heads, configuration.join_node_aggregation.count)
        self._join_edge_aggregation = MultiHeadAggregation(edge_size, configuration.join_edge_aggregation.output_size_per_head, configuration.join_edge_aggregation.num_heads, configuration.join_edge_aggregation.count)
        join_size = self._all_node_aggregation.output_size() + 2 * self._join_node_aggregation.output_size() + self._join_edge_aggregation.output_size() + encoder.join_operator_information_size()
        self._join_layers, join_size = build_layer_stack(join_size, configuration.join_layer_sizes, self._default_activation_function)
        self._final_join_layer = torch.nn.Linear(join_size, 1)
        if configuration.join_subquery_layer_sizes is not None:
            self._join_subgroup_layers, join_subquery_size = build_layer_stack(self._join_node_aggregation.output_size(), configuration.join_subquery_layer_sizes, self._default_activation_function)
            self._final_join_subquery_layer = torch.nn.Linear(join_subquery_size, 1)
        else:
            self._join_subgroup_layers = None
            self._final_join_subquery_layer = None
        self._output_operator_scaling = configuration.output_operator_scaling

        if configuration.learn_decomposed_costs:
            subgroup_size = self._join_node_aggregation.output_size() + encoder.requirement_information_size()
            self._subgroup_layers, subquery_size = build_layer_stack(subgroup_size, configuration.subquery_layer_sizes, self._default_activation_function)
            self._final_subgroup_layer = torch.nn.Linear(subquery_size, 1)

        scan_size = final_node_size + encoder.scan_operator_information_size()
        self._scan_layers, scan_size = build_layer_stack(scan_size, configuration.scan_layer_sizes, self._default_activation_function)
        self._final_scan_layer = torch.nn.Linear(scan_size, 1)

        self._log_min_local_cost = torch.log(torch.tensor(configuration.min_local_cost))
        self._softplus = torch.nn.Softplus()

    def adapt_join_information_size(self, join_information_size: int, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        """Grow the first edge layer to a larger join information size by adding zero weights for key columns added on demand."""
        assert len(self._edge_layers) > 0 and isinstance(self._edge_layers[0], torch.nn.Linear)
        layer = self._edge_layers[0]
        old_size = layer.in_features
        if old_size == join_information_size:
            return False
        assert join_information_size > old_size
        # The input is cat([first_column_encoding, second_column_encoding]), so zero columns are appended to both halves.
        old_half = old_size // 2

        def pad_columns(matrix: torch.Tensor) -> torch.Tensor:
            padding = torch.zeros((matrix.size(0), (join_information_size - old_size) // 2), dtype=matrix.dtype, device=matrix.device)
            return torch.cat([matrix[:, :old_half], padding, matrix[:, old_half:], padding], dim=1)

        # A new Parameter is required: resizing .data in place leaves a stale gradient accumulator behind.
        old_weight = layer.weight
        new_weight = torch.nn.Parameter(pad_columns(old_weight.data))
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group["params"] = [new_weight if param is old_weight else param for param in param_group["params"]]
            state = optimizer.state.pop(old_weight, None)
            if state is not None:
                for key in ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]:
                    if key in state and torch.is_tensor(state[key]) and state[key].dim() == 2:
                        state[key] = pad_columns(state[key])
                optimizer.state[new_weight] = state
        layer.weight = new_weight
        layer.in_features = join_information_size
        return True

    def _log_forward(self, data: TopDownLearnedOptimizerData) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        nodes = data.x
        if self._table_specific_weights is not None:
            sample_weights = torch.index_select(self._table_specific_weights, 0, data.table_index)
            sample_biases = torch.index_select(self._table_specific_bias, 0, data.table_index)
            table_specific_hidden = torch.matmul(sample_weights, data.table_specific_information.unsqueeze(dim=2)).squeeze(dim=2) + sample_biases
            nodes = torch.cat([nodes, table_specific_hidden], dim=1)

        for layer in self._node_layers:
            nodes = layer(nodes)

        edges = data.join_information
        for layer in self._edge_layers:
            edges = layer(edges)
        if edges.size(0) > 0:
            edges = torch.index_add(torch.zeros((data.join_information_index.max().item() + 1, edges.size(1)), device=edges.device), 0, data.join_information_index, edges)

        nodes_list = [nodes]
        join_indptr = torch.cat([data.join_indptr, torch.tensor([data.edge_index.size(1)], device=data.join_indptr.device)], dim=0)
        for message_passing_layer in self._message_passing_layers:
            nodes = message_passing_layer(nodes, data.edge_index, join_indptr, data.join_count, edges)
            nodes_list.append(nodes)

        if self._configuration.message_passing_skip_connections:
            nodes = torch.cat(nodes_list, dim=1)

        if len(data.join_operator_subgroup_index) == 0:
            join = torch.zeros((0), device=nodes.device)
            subgroups = torch.zeros((0), device=nodes.device)
        else:
            batch_indptr = torch.cat([data.batch_indptr, torch.tensor([data.x.size(0)], device=data.batch_indptr.device)], dim=0)
            if data.batch is None:
                batch = torch.zeros((data.x.size(0),), dtype=torch.long, device=data.x.device)
            else:
                batch = data.batch
            all_nodes = self._all_node_aggregation(nodes, batch, batch_indptr, data.batch_count)

            all_nodes_join = torch.index_select(all_nodes, 0, data.join_operator_batch)
            join_subgroup_nodes = torch.index_select(nodes, 0, data.join_operator_subgroup_table_index)
            join_subgroup_nodes = torch.cat([join_subgroup_nodes, data.join_operator_subgroup_table_information], dim=1)
            join_operator_subgroup_indptr = torch.cat([data.join_operator_subgroup_indptr, torch.tensor([data.join_operator_subgroup_index.size(0)], device=data.join_operator_subgroup_indptr.device)], dim=0)
            join_subgroups = self._join_node_aggregation(join_subgroup_nodes, data.join_operator_subgroup_index, join_operator_subgroup_indptr, data.join_operator_subgroup_count)
            outer_join_subgroups = torch.index_select(join_subgroups, 0, data.join_operator_outer_subgroup_index)
            inner_join_subgroups = torch.index_select(join_subgroups, 0, data.join_operator_inner_subgroup_index)
            join_edges = torch.index_select(edges, 0, data.join_operator_join_index)
            join_operator_join_indptr = torch.cat([data.join_operator_join_indptr, torch.tensor([data.join_operator_index.size(0)], device=data.join_operator_join_indptr.device)], dim=0)
            join_edges = self._join_edge_aggregation(join_edges, data.join_operator_index, join_operator_join_indptr, data.join_operator_join_count)
            join = torch.cat([all_nodes_join, outer_join_subgroups, inner_join_subgroups, join_edges, data.join_operator_information], dim=1)
            for layer in self._join_layers:
                join = layer(join)
            join = self._final_join_layer(join).squeeze(1)
            if self._join_subgroup_layers is not None:
                for layer in self._join_subgroup_layers:
                    outer_join_subgroups = layer(outer_join_subgroups)
                    inner_join_subgroups = layer(inner_join_subgroups)
                outer_join_subquery = self._final_join_subquery_layer(outer_join_subgroups).squeeze(1)
                join = torch.logaddexp(join, outer_join_subquery)
                inner_join_subquery = self._final_join_subquery_layer(inner_join_subgroups).squeeze(1)
                join = torch.logaddexp(join, inner_join_subquery)
            if self._output_operator_scaling:
                join = join + torch.log(2 * data.join_operator_num_tables - 1)
            join = self._softplus(join) + self._log_min_local_cost

            if self._configuration.learn_decomposed_costs:
                subgroups = torch.cat([join_subgroups, data.requirement_information], dim=1)
                for layer in self._subgroup_layers:
                    subgroups = layer(subgroups)
                subgroups = self._final_subgroup_layer(subgroups).squeeze(1)
                subgroups = self._softplus(subgroups) + self._log_min_local_cost
            else:
                subgroups = None

        scan_nodes = torch.index_select(nodes, 0, data.scan_operator_index)
        scan = torch.cat([scan_nodes, data.scan_operator_information], dim=1)
        for layer in self._scan_layers:
            scan = layer(scan)
        scan = self._final_scan_layer(scan).squeeze(1)
        scan = self._softplus(scan) + self._log_min_local_cost

        return join, scan, subgroups

    def log_forward(self, data: TopDownLearnedOptimizerData) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        log_join_predictions, log_scan_predictions, log_subgroup_predictions = self._log_forward(data)
        if self._configuration.learn_decomposed_costs:
            log_outer_join_subgroup_predictions = torch.index_select(log_subgroup_predictions, 0, data.join_operator_outer_subgroup_index)
            log_inner_join_subgroup_predictions = torch.index_select(log_subgroup_predictions, 0, data.join_operator_inner_subgroup_index)
            log_join_predictions = torch.logsumexp(torch.stack([log_join_predictions, log_outer_join_subgroup_predictions, log_inner_join_subgroup_predictions], dim=0), dim=0)
        return log_join_predictions, log_scan_predictions

    def forward(self, data: TopDownLearnedOptimizerData) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        log_join_predictions, log_scan_predictions = self.log_forward(data)
        join_predictions = torch.exp(log_join_predictions)
        scan_predictions = torch.exp(log_scan_predictions)
        return join_predictions, scan_predictions

    def local_forward(self, data: TopDownLearnedOptimizerData) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Per-operator local cost predictions without the decomposed subgroup cost terms that log_forward adds."""
        log_join_predictions, log_scan_predictions, _ = self._log_forward(data)
        return torch.exp(log_join_predictions), torch.exp(log_scan_predictions)

    def loss(self, data: TopDownLearnedOptimizerData, target_data: TopDownLearnedOptimizerTargetData) -> Tuple[torch.FloatTensor, np.ndarray, np.ndarray]:
        log_join_predictions, log_scan_predictions, log_subgroup_predictions = self._log_forward(data)

        all_losses = []
        min_join_losses = torch.relu(target_data.join_operator_min_cost - log_join_predictions) ** 2
        max_join_losses = torch.relu(log_join_predictions - target_data.join_operator_max_cost) ** 2
        if self._configuration.training_query_selection_mode == TrainingQuerySelectionMode.FINITE_UPPER:
            join_losses = min_join_losses + max_join_losses
            join_losses = join_losses[target_data.join_operator_cost_mask]
        elif self._configuration.training_query_selection_mode == TrainingQuerySelectionMode.FINITE_UPPER_OR_NON_ZERO_LOWER or self._configuration.training_query_selection_mode == TrainingQuerySelectionMode.FINITE_UPPER_OR_NON_ZERO_LOWER_OR_EXECUTED:
            max_join_losses = max_join_losses[target_data.join_operator_cost_mask]
            join_losses = torch.cat([min_join_losses, max_join_losses], dim=0)
        else:
            raise NotImplementedError()
        if len(join_losses) > 0:
            all_losses.append(join_losses)
        if self._configuration.learn_decomposed_costs and target_data.join_operator_subgroup_cost is not None:
            subgroup_losses = (target_data.join_operator_subgroup_cost - log_subgroup_predictions) ** 2
            if len(subgroup_losses) > 0:
                all_losses.append(subgroup_losses)
        scan_losses = torch.relu(target_data.scan_operator_min_cost - log_scan_predictions) ** 2 + torch.relu(log_scan_predictions - target_data.scan_operator_max_cost) ** 2
        scan_losses = scan_losses[target_data.scan_operator_cost_mask]
        if len(scan_losses) > 0:
            all_losses.append(scan_losses)
        if self._configuration.recursive_cost_constraint_factor is not None:
            join_constraint_losses = torch.relu(target_data.join_operator_min_cost - log_join_predictions) ** 2
            join_constraint_losses = join_constraint_losses[torch.logical_not(target_data.join_operator_cost_mask)] * self._configuration.recursive_cost_constraint_factor
            if len(join_constraint_losses) > 0:
                all_losses.append(join_constraint_losses)
        all_losses = torch.cat(all_losses, dim=0)
        loss = torch.mean(all_losses)
        return loss, log_join_predictions.detach().cpu().numpy(), log_scan_predictions.detach().cpu().numpy()







