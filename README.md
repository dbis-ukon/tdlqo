# Top-Down Learned Query Optimization (TDLQO)

Reference implementation of the paper **"Top-Down Learned Query Optimization"** by Silvan Reiner and Michael Grossniklaus (University of Konstanz).

Classical cost-based query optimizers follow either a bottom-up or a top-down framework, while published learned query optimizers (LQOs) have so far all been bottom-up. TDLQO translates query optimization into a Markov decision process over Cascades-style **groups** and **multiexpressions**, yielding a learned optimizer whose plan choices satisfy Bellman's principle of optimality: identical subqueries are optimized identically regardless of surrounding context. The framework further introduces **target decomposition** (assigning learned costs to individual operators) and **cardinality deduction** (deriving cardinality bounds across related subplan queries), which together let the model be trained from small amounts of data collected during normal database use. In our evaluation, TDLQO is the only learned approach that consistently improves on PostgreSQL across JOB, CEB-IMDb-unique, and CEB-stack.

## Repository layout

| Path | Contents                                                                                                                             |
| --- |--------------------------------------------------------------------------------------------------------------------------------------|
| `optimizers/top_down_learned_optimizer/` | The TDLQO optimizer: encoder, neural network module, configuration, and search.                                                      |
| `optimizers/` | Query optimizers.                                                                                                                    |
| `relational_algebra_expressions/` | Plan representation: groups, multiexpressions, join and scan expressions, requirements.                                              |
| `cardinality_estimators/` | Cardinality estimators incl. PostgreSQL, prophetic (true), and the cardinality-interval estimator.                                   |
| `cost_models/` | Cost models incl. the PostgreSQL cost model.                                                                                         |
| `execution_engines/` | PostgreSQL execution wrappers; `pg_hint_plan_execution_engine.py` is used to force learned plans via `pg_hint_plan`.                 |
| `neural_network_modules/` | Message passing, aggregation, and utility modules used by the model architecture.                                                    |
| `sample_encoders/` | Sample bitmap encoder for per-table predicates.                                                                                      |
| `queries/` | Query representation (SPJ queries, predicates) and benchmark loaders (JOB, JOB-light, JOB-extended, CEB-IMDb, CEB-stack, STATS-CEB). |
| `schemas/` | Schema reflection from PostgreSQL; benchmark schema entry points (`imdb_schema`, `stack_schema`, ...).                               |
| `experiments/` | Online experiments, the consistency experiment, and PostgreSQL/true-cardinality baselines.                                           |
| `tests/` | Correctness tests for cardinality bounds, batching, determinism, and the PostgreSQL cost model.                                      |
| `main.py` | Example entry point that runs an online-training experiment of TDLQO on the CEB-stack benchmark.                                     |

## Requirements

### System

- **PostgreSQL 18.0** with the **`pg_hint_plan`** extension installed and loadable.
- Benchmark databases loaded into PostgreSQL: **IMDb** (for JOB / CEB-IMDb-unique) and **StackExchange** (for CEB-stack).

### Python

- Python 3.10
- PyTorch, `torch_geometric`, `torch_scatter`
- `psycopg2`, `numpy`, `prettytable`, `binpacking`, `frozenlist`
- `wandb` (optional; `main.py` disables it via `WANDB_MODE=disabled`)

A virtual environment is set up under `venv/`. To recreate it:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install torch torch_geometric torch_scatter psycopg2-binary numpy prettytable binpacking frozenlist wandb
```

## Configuration

The code connects to PostgreSQL via `psycopg2` on `localhost`. The connection user is hard-coded to the local user in a few places (see `queries/query_data/query_db.py` and `schemas/schema.py`). Adjust the user / port to match your setup before running.

The example in `main.py` expects:
- a PostgreSQL instance on port `5443`,
- a `query_db` database containing the benchmark queries,
- a `stack` database with the StackExchange dataset and the `pg_hint_plan` extension available.

## Running the example

```bash
source venv/bin/activate
python main.py
```

This loads the CEB-stack benchmark (`benchmark_id=22`), parses the queries against the schema, and runs an `OnlineTrainingExperiment` with the JGMP-sized TDLQO configuration. `main.py` also instantiates ablation configurations used in the paper:

- `tdlqo_constructor` — full TDLQO.
- `tdlqo_ncd_constructor` — TDLQO without cardinality deduction (Section 5.2 ablation).
- `tdlqo_ndc_constructor` — TDLQO without decomposed costs / target decomposition (Section 5.1 ablation).

Swap the constructor passed to `OnlineTrainingExperiment` to reproduce the corresponding ablation curve.

## Experiments

`experiments/` contains the experiment harnesses used to generate the paper's figures:

- `online_training_experiment.py` — repeated 5-fold cross-validation with retraining after each iteration of query executions (Section 6.1, Figures 9-11).
- `consistency_experiment.py` — measures the share of subplan queries that are optimized identically standalone vs. inside their parent query (Section 6.2, Figure 12).
- `postgresql_experiment.py`, `true_cardinality_experiment.py`, `postgresql_cardinality_experiment.py` — baselines using stock PostgreSQL, true cardinalities, and PostgreSQL's cardinality estimator with our cost model.

Benchmarks supported by `queries/query_data/`: JOB, JOB-light, JOB-extended, CEB-IMDb, CEB-stack, STATS-CEB.

