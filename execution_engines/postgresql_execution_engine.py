from typing import Optional, List

from execution_engines.execution_engine import ExecutionEngine
from queries.query import Query
from queries.spj_query import SPJQuery
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression
from schemas.schema import Schema


class PostgreSQLExecutionEngine(ExecutionEngine):
    def __init__(self, schema: Schema, max_parallel_workers_per_gather: int, timeout: float = None, additional_set_commands: List[str] = []):
        super().__init__("Postgres Execution Engine", schema, additional_set_commands, max_parallel_workers_per_gather, timeout=timeout, verify_plan=False)

    def execution_query(self, query_text: str, plan: Optional[RelationalAlgebraExpression], explain_string: str) -> str:
        execution_query = explain_string + " " + query_text
        return execution_query

    def parse_plan(self, query: SPJQuery, query_text: str) -> RelationalAlgebraExpression:
        pass


