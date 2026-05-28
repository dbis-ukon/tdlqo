from enum import Enum
from typing import Dict, Optional, Tuple

from cardinality_estimators.cardinality_range import CardinalityRange
from queries.query import Query
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression


class ExecutionStatus(Enum):
    SUCCESS = 0
    TIMEOUT = 1
    INVALID = 2


class ExecutionData:
    def __init__(self,
                 query: Query,
                 query_name: Optional[str],
                 plan: Optional[RelationalAlgebraExpression],
                 executed_as_intended: Optional[bool],
                 estimated_cost: Optional[float],
                 execution_time: Optional[float],
                 planning_time: Optional[float],
                 timeout: Optional[float],
                 cardinality_ranges: Dict[Query, Tuple[CardinalityRange, bool]],
                 cardinality_estimates: Dict[Query, float],
                 explain_json: Optional[Dict]):
        self.query: Query = query
        self.query_name: Optional[str] = query_name
        self.plan: Optional[RelationalAlgebraExpression] = plan
        self.executed_as_intended: bool = executed_as_intended
        self.estimated_cost: Optional[float] = estimated_cost
        self.execution_time: Optional[float] = execution_time
        self.planning_time: Optional[float] = planning_time
        self.timeout: Optional[float] = timeout
        self.cardinality_ranges: Dict[Query, Tuple[CardinalityRange, bool]] = cardinality_ranges
        self.cardinality_estimates: Dict[Query, float] = cardinality_estimates
        self.explain_json: Optional[Dict] = explain_json

    def to_string(self) -> str:
        if self.execution_time is None:
            execution_time_str = "N/A"
        else:
            execution_time_str = f"{self.execution_time:.2f} ms"
        if self.planning_time is None:
            planning_time_str = "N/A"
        else:
            planning_time_str = f"{self.planning_time:.2f} ms"
        if self.timeout is None:
            timeout_str = "N/A"
        else:
            timeout_str = f"{self.timeout:.2f} ms"
        return "%s, %s, %s" % (execution_time_str, planning_time_str, timeout_str)

    def to_csv(self) -> str:
        if self.execution_time is None:
            execution_time_str = "-"
        else:
            execution_time_str = f"{self.execution_time}"
        if self.planning_time is None:
            planning_time_str = "-"
        else:
            planning_time_str = f"{self.planning_time}"
        if self.timeout is None:
            timeout_str = "-"
        else:
            timeout_str = f"{self.timeout}"
        return "%s,%s,%s" % (execution_time_str, planning_time_str, timeout_str)


