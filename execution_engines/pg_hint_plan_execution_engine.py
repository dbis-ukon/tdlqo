import math
from typing import List, Set, Tuple, Optional

import numpy as np

from execution_engines.execution_engine import ExecutionEngine
from queries.predicates.conjunction import Conjunction
from queries.predicates.join import Join
from queries.query import Query
from queries.spj_query import SPJQuery, SelectClauseAggregationType
from queries.table_occurrence import TableOccurrence
from relational_algebra_expressions.join_expressions.hash_join_expression import HashJoinExpression
from relational_algebra_expressions.join_expressions.join_expression import JoinExpression
from relational_algebra_expressions.join_expressions.merge_join_expression import MergeJoinExpression
from relational_algebra_expressions.join_expressions.nested_loop_join_expression import NestedLoopJoinExpression
from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression
from relational_algebra_expressions.scan_expressions.bitmap_index_scan_expression import BitmapIndexScanExpression
from relational_algebra_expressions.scan_expressions.index_scan_expression import IndexScanExpression
from relational_algebra_expressions.scan_expressions.scan_expression import ScanExpression
from relational_algebra_expressions.scan_expressions.sequential_scan_expression import SequentialScanExpression
from schemas.column import Column
from schemas.index import Index
from schemas.schema import Schema


class PgHintPlanExecutionEngine(ExecutionEngine):
    def __init__(self, schema: Schema, max_parallel_workers_per_gather: int, timeout: float = None, memoize_calls_factor: float = 0.0):
        set_commands = ["LOAD 'pg_hint_plan';",
                        "SET pg_hint_plan.enable_hint = on;",
                        "SET geqo = off;",
                        "SET join_collapse_limit = 1000;",
                        "SET enable_material = off;"]
        super().__init__("Pg Hint Plan Execution Engine", schema, set_commands, max_parallel_workers_per_gather, timeout=timeout)
        self._memoize_calls_factor: float = memoize_calls_factor

    def execution_query(self, query_text: str, plan: Optional[RelationalAlgebraExpression], explain_string: str) -> str:
        if plan is None:
            hint_comment = ""
        else:
            hint_comment = "/*+ %s */" % " ".join(self._build_hints(plan))
        execution_query = """%s\n%s %s""" % (hint_comment, explain_string, query_text)
        return execution_query

    def _build_hints(self, plan: RelationalAlgebraExpression) -> List[str]:
        root_query = plan.query()
        assert isinstance(root_query, SPJQuery)
        hints, _, leading_hint = self._build_hints_recursive(plan, root_query)
        leading_hint = "Leading ( %s )" % leading_hint
        return [leading_hint] + hints

    def _build_hints_recursive(self, plan: RelationalAlgebraExpression, root_query: SPJQuery) -> Tuple[List[str], List[str], str]:
        hints = []
        aliases = []
        leading_hints = []
        for child in plan.children:
            child_hints, child_aliases, child_leading_string = self._build_hints_recursive(child, root_query)
            hints.extend(child_hints)
            aliases.extend(child_aliases)
            leading_hints.append(child_leading_string)
        if isinstance(plan, ScanExpression):
            alias = plan.table_occurrence.alias()
            aliases.append(alias)
            leading_hint = alias
            if isinstance(plan, SequentialScanExpression):
                hints.append("SeqScan(%s)" % alias)
            elif isinstance(plan, IndexScanExpression):
                if plan.index is None:
                    hints.append("IndexScan(%s)" % alias)
                elif self._index_only_scan_applicable(plan.index, plan.table_occurrence, root_query):
                    hints.append("IndexOnlyScan(%s %s)" % (alias, plan.index.name()))
                else:
                    hints.append("IndexScan(%s %s)" % (alias, plan.index.name()))
            elif isinstance(plan, BitmapIndexScanExpression):
                hints.append("BitmapIndexScan(%s)" % alias)
            else:
                raise NotImplementedError()
        elif isinstance(plan, JoinExpression):
            assert len(leading_hints) == 2
            aliases_string = " ".join(aliases)
            leading_hint = "( %s %s )" % (leading_hints[0], leading_hints[1])
            if isinstance(plan, NestedLoopJoinExpression):
                hints.append("NestLoop(%s)" % aliases_string)
                if not isinstance(plan.inner, IndexScanExpression):
                    outer_query = plan.outer.query()
                    assert isinstance(outer_query, SPJQuery)
                    table_occurrences = outer_query.table_occurrences()
                    if len(table_occurrences) > 1:
                        outer_aliases = [table_occurrence.alias() for table_occurrence in table_occurrences]
                        outer_aliases_string = " ".join(outer_aliases)
                        hints.append("Rows(%s +2)" % outer_aliases_string)
                elif plan.inner.memoized:
                    memoize_rows_hint = self._memoize_rows_hint(plan)
                    if memoize_rows_hint is not None:
                        hints.append(memoize_rows_hint)
            elif isinstance(plan, HashJoinExpression):
                hints.append("HashJoin(%s)" % aliases_string)
            elif isinstance(plan, MergeJoinExpression):
                hints.append("MergeJoin(%s)" % aliases_string)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        return hints, aliases, leading_hint

    def _memoize_rows_hint(self, plan: NestedLoopJoinExpression) -> Optional[str]:
        if self._memoize_calls_factor == 0.0:
            return None
        # Skip single-table outers: pg_hint_plan's Rows(...) needs >= 2 aliases.
        outer_query = plan.outer.query()
        assert isinstance(outer_query, SPJQuery)
        outer_table_occurrences = outer_query.table_occurrences()
        if len(outer_table_occurrences) <= 1:
            return None

        inner_table_occurrence = plan.inner.table_occurrence

        join_condition = plan.join_condition
        if isinstance(join_condition, Join):
            join_clauses = [join_condition]
        elif isinstance(join_condition, Conjunction):
            join_clauses = join_condition.predicates()
        else:
            return None

        inner_memo_columns: List[Column] = []
        seen: Set[Column] = set()
        for clause in join_clauses:
            if not isinstance(clause, Join):
                continue
            for table_occurrence, column in clause.equivalence_class():
                if table_occurrence == inner_table_occurrence and column not in seen:
                    seen.add(column)
                    inner_memo_columns.append(column)
        if not inner_memo_columns:
            return None

        distinct_counts = []
        for column in inner_memo_columns:
            nd = column.distinct_count()
            if nd is None:
                return None
            distinct_counts.append(float(nd))

        loop_product = float(np.prod(distinct_counts))
        loop_max = max(distinct_counts)
        d = math.sqrt(loop_max * loop_product)
        n = math.ceil(self._memoize_calls_factor * d)

        outer_aliases_string = " ".join(table_occurrence.alias() for table_occurrence in outer_table_occurrences)
        return "Rows(%s +%d)" % (outer_aliases_string, n)

    @staticmethod
    def _index_only_scan_applicable(index: Index, table_occurrence: TableOccurrence, root_query: SPJQuery) -> bool:
        needed: Set[Column] = set()
        needed |= table_occurrence.predicate().columns_for(table_occurrence)
        needed |= root_query.join_predicate().columns_for(table_occurrence)
        for non_equi_predicate in root_query.non_equi_join_predicates():
            needed |= non_equi_predicate.columns_for(table_occurrence)
        needed |= PgHintPlanExecutionEngine._projection_columns(root_query, table_occurrence)
        return index.covers_columns(needed)

    @staticmethod
    def _projection_columns(root_query: SPJQuery, table_occurrence: TableOccurrence) -> Set[Column]:
        columns: Set[Column] = set()
        for element in root_query.select_clause_elements():
            if element.table_occurrence is None and element.column is None:
                if element.aggregation_type == SelectClauseAggregationType.COUNT:
                    continue
                columns.update(table_occurrence.table().columns())
            elif element.table_occurrence == table_occurrence:
                if element.column is None:
                    if element.aggregation_type == SelectClauseAggregationType.COUNT:
                        continue
                    columns.update(table_occurrence.table().columns())
                else:
                    columns.add(element.column)
        return columns




