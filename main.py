
import os

os.environ["WANDB_MODE"] = "disabled"

from cardinality_estimators.postgresql_cardinality_estimator import PostgreSQLCardinalityEstimator
from cardinality_estimators.prophetic_cardinality_estimator import PropheticCardinalityEstimator
from cost_models.postgresql_cost_model import PostgreSQLCostModel
from execution_engines.pg_hint_plan_execution_engine import PgHintPlanExecutionEngine
from experiments.consistency_experiment import ConsistencyExperiment
from experiments.experiment import Experiment
from experiments.memoized_expression_count_experiment import MemoizedExpressionCountExperiment
from experiments.multi_optimizer_experiment import MultiOptimizerExperiment
from experiments.online_training_experiment import OnlineTrainingExperiment
from experiments.postgresql_cardinality_experiment import PostgreSQLCardinalityExperiment
from experiments.postgresql_experiment import PostgreSQLExperiment
from experiments.regret_decomposition_experiment import RegretDecompositionExperiment
from experiments.simple_experiment import SimpleExperiment
from experiments.true_cardinality_experiment import TrueCardinalityExperiment
from optimizers.top_down_cost_based_optimizer import TopDownCostBasedOptimizer
from optimizers.top_down_learned_optimizer.tdlqo_learned_cost_model import TDLQOLearnedCostModel
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer import TopDownLearnedOptimizer
from optimizers.top_down_learned_optimizer.top_down_learned_optimizer_configuration import TopDownLearnedOptimizerConfiguration
from optimizers.top_down_random_optimizer import TopDownRandomOptimizer
from queries.query_data.load_benchmark import load_benchmark
from queries.query_data.parse import parse_spj_queries
from queries.query_data.query_db import QueryDB
from queries.spj_query import SPJQuery
from schemas.benchmark_schemas import imdb_schema, stack_schema

query_db = QueryDB("query_db", 5443)
schema = stack_schema(port=5443)
benchmark_queries = load_benchmark(query_db, 22)
parse_spj_queries(schema, benchmark_queries, verify_correctness=False)
benchmark_queries = [benchmark_query for benchmark_query in benchmark_queries if benchmark_query.query is not None]
timeout = 2 * 60 * 1000
execution_engine = PgHintPlanExecutionEngine(schema, 2, timeout=timeout)

tdlqo_constructor = lambda all_queries, training_queries: TopDownLearnedOptimizer(schema, TopDownLearnedOptimizerConfiguration.default_configuration(schema))

tdlqo_ncd_constructor = lambda all_queries, training_queries: TopDownLearnedOptimizer(schema, TopDownLearnedOptimizerConfiguration.no_cardinality_deduction_configuration(schema))
tdlqo_ndc_constructor = lambda all_queries, training_queries: TopDownLearnedOptimizer(schema, TopDownLearnedOptimizerConfiguration.no_decomposed_costs_configuration(schema))


online_training_experiment = OnlineTrainingExperiment(tdlqo_constructor, execution_engine, 20, benchmark_queries, 5, 3, True, [])
online_training_experiment.run()


def job_complex_experiment():
    # JOB-Complex (benchmark 25) with a fixed test set: every repetition trains on the JOB-Complex-Train queries
    # (benchmark 26, three predicate variants per JOB-Complex query, see
    # queries/query_data/generate_job_complex_training.py) and tests on the JOB-Complex queries.
    job_complex_schema = imdb_schema(port=5443)
    job_complex_queries = load_benchmark(query_db, 25)
    parse_spj_queries(job_complex_schema, job_complex_queries, verify_correctness=False)
    job_complex_queries = [benchmark_query for benchmark_query in job_complex_queries if benchmark_query.query is not None]
    training_queries = load_benchmark(query_db, 26)
    parse_spj_queries(job_complex_schema, training_queries, verify_correctness=False)
    training_queries = [benchmark_query for benchmark_query in training_queries if benchmark_query.query is not None]
    job_complex_execution_engine = PgHintPlanExecutionEngine(job_complex_schema, 2, timeout=timeout)
    constructor = lambda all_queries, training_queries: TopDownLearnedOptimizer(job_complex_schema, TopDownLearnedOptimizerConfiguration.default_configuration(job_complex_schema))
    experiment = OnlineTrainingExperiment(constructor, job_complex_execution_engine, 20, training_queries, None, 3, True, [], test_queries=job_complex_queries)
    experiment.run()


#job_complex_experiment()


def memoized_expression_count_experiment():
    experiment = MemoizedExpressionCountExperiment(TopDownRandomOptimizer(TopDownLearnedOptimizerConfiguration.default_configuration(schema).available_operators), benchmark_queries, 10)
    experiment.run()


#memoized_expression_count_experiment()


def performance_pipeline_experiments(tdlqo_run_path: str, num_splits: int):
    # Three cost-based optimizers over the same search space, on JOB (benchmark 2, IMDb):
    # the teacher cost model with PostgreSQL cardinality estimates, the teacher cost model with
    # true cardinalities, and the learned operator-level cost model of a TDLQO training run.
    # tdlqo_run_path is the result directory of an OnlineTrainingExperiment on JOB whose
    # checkpoints optimizer_iteration_<split>_20 are loaded for the first num_splits splits.
    job_schema = imdb_schema(port=5443)
    job_benchmark_queries = load_benchmark(query_db, 2)
    parse_spj_queries(job_schema, job_benchmark_queries, verify_correctness=False)
    job_benchmark_queries = [benchmark_query for benchmark_query in job_benchmark_queries if benchmark_query.query is not None]
    job_execution_engine = PgHintPlanExecutionEngine(job_schema, 2, timeout=timeout)

    # Test sets of the cross-validation splits of the training run the learned cost models are loaded from.
    splits = Experiment._cross_validation_splits(job_benchmark_queries, 5, 3)
    test_sets = [splits[i][1] for i in range(num_splits)]
    teacher_configuration = TopDownLearnedOptimizerConfiguration.default_configuration(job_schema)
    available_operators = teacher_configuration.available_operators

    # Setup 1: teacher cost model with PostgreSQL cardinality estimates
    cardinality_estimator = PostgreSQLCardinalityEstimator(job_schema, True)
    cost_model = teacher_configuration.cost_model.replace_cardinality_estimator(cardinality_estimator)
    optimizer = TopDownCostBasedOptimizer(cost_model, available_operators)
    postgresql_cardinality_experiment = SimpleExperiment(optimizer, job_execution_engine, [job_benchmark_queries], repeated_executions=3)
    postgresql_cardinality_experiment.run()

    # Setup 2: teacher cost model with true cardinalities
    cardinality_estimator = PropheticCardinalityEstimator(gather_on_demand_schema=job_schema, gather_on_demand_timeout_seconds=1)
    cardinality_estimator.load_cardinalities("queries/query_data/job_true_cardinalities.json", job_benchmark_queries)
    cost_model = teacher_configuration.cost_model.replace_cardinality_estimator(cardinality_estimator)
    optimizer = TopDownCostBasedOptimizer(cost_model, available_operators)
    true_cardinality_experiment = SimpleExperiment(optimizer, job_execution_engine, [job_benchmark_queries], repeated_executions=3)
    true_cardinality_experiment.run()

    # Setup 3: learned operator-level cost model without forward-looking subgroup cost estimates
    optimizers = []
    for i in range(num_splits):
        cost_model = TDLQOLearnedCostModel(TopDownLearnedOptimizerConfiguration.default_configuration(job_schema), job_schema, tdlqo_run_path + "/optimizer_iteration_%d_20" % i)
        optimizers.append(TopDownCostBasedOptimizer(cost_model, available_operators))
    learned_cost_model_experiment = MultiOptimizerExperiment(optimizers, job_execution_engine, test_sets)
    learned_cost_model_experiment.run()


#performance_pipeline_experiments("results/<tdlqo training run on JOB>", 15)


def regret_decomposition_experiment(tdlqo_run_path: str, splits: tuple = (0, 1, 2, 3, 4), iterations: int = 20):
    # Regret of the learned optimizer against the teacher cost model on true cardinalities: for every
    # group in the search space of the held-out queries, how much the chosen multiexpression costs
    # above the optimum, split into join order and physical implementation. Splits 0-4 form one
    # complete five fold partition of the benchmark.
    job_schema = imdb_schema(port=5443)
    job_benchmark_queries = load_benchmark(query_db, 2)
    parse_spj_queries(job_schema, job_benchmark_queries, verify_correctness=False)
    job_benchmark_queries = [benchmark_query for benchmark_query in job_benchmark_queries if benchmark_query.query is not None]
    cross_validation_splits = Experiment._cross_validation_splits(job_benchmark_queries, 5, 3)

    # The exported JSON holds the true cardinality of every group in the search space of the JOB
    # queries, so no cardinalities are gathered on demand.
    cardinality_estimator = PropheticCardinalityEstimator()
    cardinality_estimator.load_cardinalities("queries/query_data/job_true_cardinalities.json", job_benchmark_queries)
    teacher_cost_model = TopDownLearnedOptimizerConfiguration.default_configuration(job_schema).cost_model.replace_cardinality_estimator(cardinality_estimator)

    configuration = TopDownLearnedOptimizerConfiguration.default_configuration(job_schema)
    split_configurations = []
    for split in splits:
        checkpoints = [(iteration, tdlqo_run_path + "/optimizer_iteration_%d_%d" % (split, iteration))
                       for iteration in range(1, iterations + 1)]
        split_configurations.append((split, checkpoints, cross_validation_splits[split][1]))
    experiment = RegretDecompositionExperiment(job_schema, configuration, teacher_cost_model, split_configurations)
    experiment.run()


#regret_decomposition_experiment("results/<tdlqo training run on JOB>")


def target_violation_experiment():
    # How often does the cost interval a training target is built from miss the cost under the true
    # cardinalities, and does that affect plan quality? Both runs carry the true cardinality cost
    # model, which records the violation statistics of each training round in the training meta
    # data (true_cardinality_*_violation_* and true_cardinality_lower_only_violation_*). The
    # corrected run additionally widens every violated interval to contain the true cost.
    job_schema = imdb_schema(port=5443)
    job_benchmark_queries = load_benchmark(query_db, 2)
    parse_spj_queries(job_schema, job_benchmark_queries, verify_correctness=False)
    job_benchmark_queries = [benchmark_query for benchmark_query in job_benchmark_queries if benchmark_query.query is not None]
    job_execution_engine = PgHintPlanExecutionEngine(job_schema, 2, timeout=timeout)

    true_cardinality_estimator = PropheticCardinalityEstimator()
    true_cardinality_estimator.load_cardinalities("queries/query_data/job_true_cardinalities.json", job_benchmark_queries)

    def oracle_constructor(configuration_factory):
        # Configurations hold the cardinality memory, so every optimizer needs its own.
        def construct(all_queries, training_queries):
            optimizer = TopDownLearnedOptimizer(job_schema, configuration_factory(job_schema))
            optimizer.set_true_cardinality_cost_model(true_cardinality_estimator)
            return optimizer
        return construct

    uncorrected_experiment = OnlineTrainingExperiment(oracle_constructor(TopDownLearnedOptimizerConfiguration.default_configuration), job_execution_engine, 20, job_benchmark_queries, 5, 3, True, [])
    uncorrected_experiment.run()
    corrected_experiment = OnlineTrainingExperiment(oracle_constructor(TopDownLearnedOptimizerConfiguration.corrected_targets_configuration), job_execution_engine, 20, job_benchmark_queries, 5, 3, True, [])
    corrected_experiment.run()


#target_violation_experiment()
