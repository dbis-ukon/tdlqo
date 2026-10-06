
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Type, List

from cardinality_estimators.cardinality_estimator import CardinalityEstimator
from cardinality_estimators.postgresql_cardinality_estimator import PostgreSQLCardinalityEstimator
from cardinality_estimators.prophetic_cardinality_estimator import PropheticCardinalityEstimator
from cost_models.c_out_cost_model import COutCostModel
from cost_models.c_outer_cost_model import COuterCostModel
from cost_models.local_cost_model import LocalCostModel
from cost_models.postgresql_cost_model import PostgreSQLCostModel
from relational_algebra_expressions.join_expressions.hash_join_expression import HashJoinExpression
from relational_algebra_expressions.join_expressions.merge_join_expression import MergeJoinExpression
from relational_algebra_expressions.join_expressions.nested_loop_join_expression import NestedLoopJoinExpression
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression
from relational_algebra_expressions.scan_expressions.index_scan_expression import IndexScanExpression
from relational_algebra_expressions.scan_expressions.sequential_scan_expression import SequentialScanExpression
from schemas.schema import Schema


@dataclass
class MessagePassingLayerConfiguration:
    message_size: int
    num_heads: int
    head_size: int
    output_size: int
    count: bool


class MultiHeadAggregationConfiguration:
    def __init__(self, output_size_per_head: int, num_heads: int, count: bool):
        self.output_size_per_head = output_size_per_head
        self.num_heads = num_heads
        self.count = count


class ActivationFunction(Enum):
    RELU = 0
    LEAKY_RELU = 1
    PRELU = 2


class TrainingQuerySelectionMode(Enum):
    FINITE_UPPER = 0
    FINITE_UPPER_OR_NON_ZERO_LOWER = 1
    FINITE_UPPER_OR_NON_ZERO_LOWER_OR_EXECUTED = 2


class TopDownLearnedOptimizerConfiguration:
    def __init__(self,
                 name: str,
                 description: str,
                 available_operators: List[Type[RelationalAlgebraExpression]],
                 cost_model: LocalCostModel,
                 untrained_postgresql: bool,
                 workers: int,
                 num_training_epochs: int,
                 initial_training_epochs: Optional[int],
                 training_epoch_decay: Optional[float],
                 max_batch_size: int,
                 target_batch_size: int,
                 learning_rate: float,
                 weight_decay: float,
                 double_q_learning: bool,
                 deduce_cardinality_ranges: bool,
                 training_query_selection_mode: TrainingQuerySelectionMode,
                 fully_explored_supervised: bool,
                 recursive_cost_constraint_factor: Optional[float],
                 pre_training_seconds: int,
                 exploration_timeout_seconds: Optional[int],
                 learn_decomposed_costs: bool,
                 min_local_cost: float,
                 table_occurrence_encoding_cardinality_estimator: Optional[CardinalityEstimator],
                 table_occurrence_encoding_predicate_columns: bool,
                 table_occurrence_encoding_predicate_costs: bool,
                 table_occurrence_encoding_index_viable: bool,
                 table_occurrence_encoding_random_sample: int,
                 table_occurrence_encoding_feature_selected_sample: int,
                 join_operator_table_occurrence_encoding_neighborhood: bool,
                 table_specific_hidden_size: int,
                 default_activation_function: ActivationFunction,
                 node_layer_sizes: List[int],
                 edge_layer_sizes: List[int],
                 message_passing_layer_configurations: List[MessagePassingLayerConfiguration],
                 message_passing_skip_connections: bool,
                 all_node_aggregation: MultiHeadAggregationConfiguration,
                 join_node_aggregation: MultiHeadAggregationConfiguration,
                 join_edge_aggregation: MultiHeadAggregationConfiguration,
                 join_layer_sizes: List[int],
                 join_subquery_layer_sizes: Optional[List[int]],
                 subquery_layer_sizes: Optional[List[int]],
                 scan_layer_sizes: List[int],
                 output_operator_scaling: Optional[bool],
                 probe_min_cardinality_threshold: Optional[int],
                 probe_max_min_ratio_threshold: float,
                 probe_per_probe_time_budget: float,
                 probe_fetch_size: int,
                 composite_max_degree_scaling: bool,
                 correct_target_violations: bool = False):
        self.name = name
        self.description = description
        self.cost_model = cost_model
        self.untrained_postgresql = untrained_postgresql
        self.workers = workers
        self.num_training_epochs = num_training_epochs
        self.initial_training_epochs = initial_training_epochs
        self.training_epoch_decay = training_epoch_decay
        self.max_batch_size = max_batch_size
        self.target_batch_size = target_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.double_q_learning = double_q_learning
        self.deduce_cardinality_ranges = deduce_cardinality_ranges
        self.training_query_selection_mode = training_query_selection_mode
        self.fully_explored_supervised = fully_explored_supervised
        self.recursive_cost_constraint_factor = recursive_cost_constraint_factor
        self.pre_training_seconds = pre_training_seconds
        self.exploration_timeout_seconds = exploration_timeout_seconds
        self.learn_decomposed_costs = learn_decomposed_costs
        self.min_local_cost = min_local_cost
        self.table_occurrence_encoding_cardinality_estimator = table_occurrence_encoding_cardinality_estimator
        self.table_occurrence_encoding_predicate_columns = table_occurrence_encoding_predicate_columns
        self.table_occurrence_encoding_predicate_costs = table_occurrence_encoding_predicate_costs
        self.table_occurrence_encoding_index_viable = table_occurrence_encoding_index_viable
        self.table_occurrence_encoding_random_sample = table_occurrence_encoding_random_sample
        self.table_occurrence_encoding_feature_selected_sample = table_occurrence_encoding_feature_selected_sample
        self.join_operator_table_occurrence_encoding_neighborhood = join_operator_table_occurrence_encoding_neighborhood
        self.table_specific_hidden_size = table_specific_hidden_size
        self.available_operators = available_operators
        self.default_activation_function = default_activation_function
        self.node_layer_sizes = node_layer_sizes
        self.edge_layer_sizes = edge_layer_sizes
        self.message_passing_layer_configurations = message_passing_layer_configurations
        self.message_passing_skip_connections = message_passing_skip_connections
        self.all_node_aggregation = all_node_aggregation
        self.join_node_aggregation = join_node_aggregation
        self.join_edge_aggregation = join_edge_aggregation
        self.join_layer_sizes = join_layer_sizes
        self.join_subquery_layer_sizes = join_subquery_layer_sizes
        self.subquery_layer_sizes = subquery_layer_sizes
        self.scan_layer_sizes = scan_layer_sizes
        self.output_operator_scaling = output_operator_scaling
        self.probe_min_cardinality_threshold = probe_min_cardinality_threshold
        self.probe_max_min_ratio_threshold = probe_max_min_ratio_threshold
        self.probe_per_probe_time_budget = probe_per_probe_time_budget
        self.probe_fetch_size = probe_fetch_size
        self.composite_max_degree_scaling = composite_max_degree_scaling
        # Widen every training target interval that does not contain the local cost under the true
        # cardinalities just enough to contain it. Needs the true cardinality cost model of the optimizer.
        self.correct_target_violations = correct_target_violations

    @staticmethod
    def default_configuration(schema: Schema) -> TopDownLearnedOptimizerConfiguration:
        return TopDownLearnedOptimizerConfiguration("Default",
                                                    "Default configuration",
                                                    [HashJoinExpression, NestedLoopJoinExpression, SequentialScanExpression, IndexScanExpression],
                                                    PostgreSQLCostModel(PropheticCardinalityEstimator(), schema.postgresql_configuration(), schema=schema, default_width=8, improved=True, index_scan_base_cost=1, probe_cache_penalty=2),
                                                    True,
                                                    8,
                                                    200,
                                                    1000,
                                                    0.5,
                                                    1024,
                                                    1024,
                                                    5e-5,
                                                    0,
                                                    False,
                                                    True,
                                                    TrainingQuerySelectionMode.FINITE_UPPER_OR_NON_ZERO_LOWER_OR_EXECUTED,
                                                    False,
                                                    None,
                                                    0,
                                                    60,
                                                    True,
                                                    1,
                                                    PostgreSQLCardinalityEstimator(schema, False),
                                                    False,
                                                    True,
                                                    True,
                                                    0,
                                                    16,
                                                    True,
                                                    32,
                                                    ActivationFunction.PRELU,
                                                    [64],
                                                    [8],
                                                    [MessagePassingLayerConfiguration(32, 4, 8, 32, True), MessagePassingLayerConfiguration(32, 4, 8, 32, True), MessagePassingLayerConfiguration(32, 4, 8, 32, True)],
                                                    True,
                                                    MultiHeadAggregationConfiguration(128, 4, True),
                                                    MultiHeadAggregationConfiguration(128, 4, True),
                                                    MultiHeadAggregationConfiguration(8, 2, True),
                                                    [256],
                                                    None,
                                                    [256],
                                                    [64],
                                                    False,
                                                    None,
                                                    2.0,
                                                    60.0,
                                                    10000,
                                                    False)

    @staticmethod
    def no_cardinality_deduction_configuration(schema: Schema) -> TopDownLearnedOptimizerConfiguration:
        configuration = TopDownLearnedOptimizerConfiguration.default_configuration(schema)
        configuration.deduce_cardinality_ranges = False
        return configuration

    @staticmethod
    def no_decomposed_costs_configuration(schema: Schema) -> TopDownLearnedOptimizerConfiguration:
        configuration = TopDownLearnedOptimizerConfiguration.default_configuration(schema)
        configuration.learn_decomposed_costs = False
        configuration.learning_rate = configuration.learning_rate / 100  # Necessary to stabilize training without decomposed costs, otherwise we see divergence in training
        return configuration

    @staticmethod
    def corrected_targets_configuration(schema: Schema) -> TopDownLearnedOptimizerConfiguration:
        configuration = TopDownLearnedOptimizerConfiguration.default_configuration(schema)
        configuration.correct_target_violations = True
        return configuration
