import csv
import re
from decimal import Decimal
from typing import List, Optional, Tuple, Dict, Any, Iterable

from sqlglot.dialects import Postgres

from queries.predicates.conjunction import Conjunction
from queries.predicates.disjunction import Disjunction
from queries.predicates.join import Join
from queries.predicates.literals.null_literal import NullLiteral
from queries.predicates.literals.number_literal import NumberLiteral
from queries.predicates.literals.tuple_literal import TupleLiteral
from queries.predicates.negation import Negation
from queries.table_occurrence import TableOccurrence
from queries.benchmark_query import BenchmarkQuery
from queries.predicates.comparison_predicate import ComparisonPredicate
from queries.predicates.comparison_operator import COMPARISON_OPERATOR_EQ, COMPARISON_OPERATOR_GT, \
    COMPARISON_OPERATOR_LT, COMPARISON_OPERATOR_LIKE, COMPARISON_OPERATOR_ILIKE, COMPARISON_OPERATOR_GTE, \
    COMPARISON_OPERATOR_LTE, COMPARISON_OPERATOR_IS, COMPARISON_OPERATOR_NEQ, COMPARISON_OPERATOR_IN, \
    COMPARISON_OPERATOR_REGEX
from queries.predicates.literals.string_literal import StringLiteral
from queries.predicates.predicate import Predicate
from queries.predicates.simple_predicate import SimplePredicate
from queries.predicates.true_predicate import TruePredicate
from queries.spj_query import SPJQuery, SelectClauseElement, SelectClauseAggregationType
from schemas.column import Column
from schemas.data_types.data_type import DataType
from schemas.data_types.numeric_data_type import NumericDataType
from schemas.data_types.postgresql_data_types import get_postgresql_data_type
from schemas.data_types.string_data_type import StringDataType
from schemas.schema import Schema

import sqlglot

from schemas.table import Table
from queries.predicates.literals.literal import Literal


def parse_spj_queries(schema: Schema, benchmark_queries: List[BenchmarkQuery], verify_correctness: bool = True):
    failed_count = 0
    for benchmark_query in benchmark_queries:
        try:
            query = parse_spj_query(schema, benchmark_query, verify_correctness=verify_correctness)
            benchmark_query.query = query
            print("Parsed query: " + benchmark_query.query_name())
        except:
            print("Failed to parse query: " + benchmark_query.query_name())
            failed_count += 1
    print("Failed to parse %d out of %d queries" % (failed_count, len(benchmark_queries)))

def verify_equality(schema: Schema, benchmark_queries: List[BenchmarkQuery]):
    for benchmark_query in benchmark_queries:
        first_parse = parse_spj_query(schema, benchmark_query, verify_correctness=False)
        second_parse = parse_spj_query(schema, benchmark_query, verify_correctness=False)
        first_known_equal = first_parse.known_equal(second_parse)
        second_known_equal = second_parse.known_equal(first_parse)
        print(benchmark_query.query_name(), first_known_equal, second_known_equal)

def parse_spj_query(schema: Schema, benchmark_query: BenchmarkQuery, verify_correctness: bool = True) -> Optional[SPJQuery]:
    connection = schema.connection()
    cursor = connection.cursor()
    cursor.execute("SET enable_indexscan = off;")
    cursor.execute("SET enable_bitmapscan = off;")
    cursor.execute("SET max_parallel_workers_per_gather = 0;")
    cursor.execute("EXPLAIN (ANALYZE FALSE, FORMAT JSON, VERBOSE TRUE) " + benchmark_query.query_text)
    explain = cursor.fetchone()[0][0]
    cursor.close()
    result = _build_pg_explain_query(schema, explain["Plan"])
    if result is None:
        print("Failed to parse query: " + benchmark_query.query_text)
        return None
    query, pushed_down_predicates, _ = result
    assert len(pushed_down_predicates) == 0
    _duplicate_predicates_across_joins(query.table_occurrences(), query.joins())
    select_clause_elements = _get_select_clause_elements(query.table_occurrences(), explain["Plan"])
    query = SPJQuery(query.table_occurrences(), query.joins(), query.non_equi_join_predicates(), select_clause_elements=select_clause_elements)
    query.query_name = benchmark_query.query_name()
    if verify_correctness:
        assert _verify_parsed_query(schema, benchmark_query, query)
    return query

def _get_select_clause_elements(table_occurrences: Iterable[TableOccurrence], explain_node: Dict[str, Any]) -> Optional[List[SelectClauseElement]]:
    table_occurrence_dict = {to.alias() if to.alias() is not None else to.table().name(): to for to in table_occurrences}
    select_clause_elements = []
    if "Output" not in explain_node:
        raise ValueError("No Output in explain node")
    for output in explain_node["Output"]:
        if output.startswith("(") and output.endswith(")"):
            output = output[1:-1]
        if output.startswith("min(") and output.endswith(")"):
            output = output[4:-1]
            select_clause_aggregation_type = SelectClauseAggregationType.MIN
        elif output.startswith("max(") and output.endswith(")"):
            output = output[4:-1]
            select_clause_aggregation_type = SelectClauseAggregationType.MAX
        elif output.startswith("count(") and output.endswith(")"):
            output = output[6:-1]
            select_clause_aggregation_type = SelectClauseAggregationType.COUNT
        else:
            # TODO: I know, I know, SQL can do a lot more... I am but one man
            select_clause_aggregation_type = SelectClauseAggregationType.NONE
        if output.startswith("(") and output.endswith(")::text"):
            output = output[1:-7]
        if output == "*":
            table_occurrence = None
            column = None
        elif len(table_occurrences) == 1:
            table_occurrence = next(iter(table_occurrences))
            column_name = output
            column = table_occurrence.table().column(column_name)
        else:
            assert "." in output
            table_alias, column_name = output.split(".", 1)
            table_occurrence = table_occurrence_dict[table_alias]
            column = table_occurrence.table().column(column_name)
        select_clause_element = SelectClauseElement(table_occurrence, column, select_clause_aggregation_type)
        select_clause_elements.append(select_clause_element)
    return select_clause_elements

def _verify_parsed_query(schema: Schema, benchmark_query: BenchmarkQuery, query: SPJQuery) -> bool:
    benchmark_query_text = re.sub(r"(?i)select\s+.*?(?=\bfrom\b)", "SELECT COUNT(*) ", benchmark_query.query_text, count=1, flags=re.DOTALL)
    # remove group by and order by clauses
    benchmark_query_text = re.sub(r"(?i)\bgroup\s+by\b.*?(?=(\border\s+by\b|;|$))", "", benchmark_query_text, flags=re.DOTALL)
    benchmark_query_text = re.sub(r"(?i)\border\s+by\b.*?(?=;|$)", "", benchmark_query_text, flags=re.DOTALL)
    print("Verifying parsed query %s with id %d" % (benchmark_query.query_name(), benchmark_query.query_id))
    print(benchmark_query_text)
    parsed_query_text = query.get_query_text("COUNT(*)")
    print(parsed_query_text)
    print()
    cursor = schema.connection().cursor()
    cursor.execute("SET max_parallel_workers_per_gather = 0;")
    cursor.execute(benchmark_query_text)
    expected = cursor.fetchone()[0]
    cursor.execute(parsed_query_text)
    actual = cursor.fetchone()[0]
    cursor.close()
    if expected != actual:
        print("Expected: " + str(expected))
        print("Actual: " + str(actual))
        return False
    print("Verified: " + str(expected))
    print()
    print()
    return True

def _build_pg_explain_query(schema: Schema, pg_explain: dict) -> Optional[Tuple[SPJQuery, List[str], Dict[float, List[Tuple[TableOccurrence, Column]]]]]:
    if "Plans" in pg_explain:
        num_plans = len(pg_explain["Plans"])
        if num_plans == 1 and pg_explain["Node Type"] == "Bitmap Heap Scan":
            return _build_scan_query(schema, pg_explain)
        elif num_plans == 1:
            return _build_pg_explain_query(schema, pg_explain["Plans"][0])
        elif num_plans == 2 and ("Join" in pg_explain["Node Type"] or pg_explain["Node Type"] == "Nested Loop"):
            subplan_queries = []
            new_join_conditions = []
            equality_dict = {}
            for subplan in pg_explain["Plans"]:
                result = _build_pg_explain_query(schema, subplan)
                if result is None:
                    return None
                subplan_query, pushed_down_predicates, subplan_equality_dict = result
                subplan_queries.append(subplan_query)
                new_join_conditions.extend(pushed_down_predicates)
                for value, pairs in subplan_equality_dict.items():
                    if value not in equality_dict:
                        equality_dict[value] = []
                    equality_dict[value].extend(pairs)
            table_occurrences = []
            joins = []
            non_equi_join_predicates = []
            alias_dict = {}
            for subplan_query in subplan_queries:
                table_occurrences.extend(subplan_query.table_occurrences())
                joins.extend(subplan_query.joins())
                non_equi_join_predicates.extend(subplan_query.non_equi_join_predicates())
                for table_occurrence in subplan_query.table_occurrences():
                    alias = table_occurrence.alias()
                    if alias is None:
                        alias = table_occurrence.table().name()
                    assert alias not in alias_dict
                    alias_dict[alias] = table_occurrence
            if "Hash Cond" in pg_explain:
                new_join_conditions.append(pg_explain["Hash Cond"])
            if "Join Filter" in pg_explain:
                new_join_conditions.append(pg_explain["Join Filter"])
            if "Merge Cond" in pg_explain:
                new_join_conditions.append(pg_explain["Merge Cond"])
            new_joins = []
            for new_join_condition in new_join_conditions:
                if " AND " in new_join_condition:
                    atomic_join_conditions = new_join_condition[1:-1].split(" AND ")
                else:
                    atomic_join_conditions = [new_join_condition]
                for atomic_join_condition in atomic_join_conditions:
                    left_table_alias, left_column_name, right_table_alias, right_column_name = _parse_equi_join_condition(atomic_join_condition)
                    left_table_occurrence = alias_dict[left_table_alias]
                    left_column = left_table_occurrence.table().column(left_column_name)
                    right_table_occurrence = alias_dict[right_table_alias]
                    right_column = right_table_occurrence.table().column(right_column_name)
                    join = Join([(left_table_occurrence, left_column), (right_table_occurrence, right_column)])
                    new_joins.append(join)
            if len(new_joins) == 0:
                outer_table_occurrences = subplan_queries[0].table_occurrences()
                for value, pairs in equality_dict.items():
                    if len(pairs) < 2:
                        continue
                    outer_pairs = []
                    inner_pairs = []
                    for table_occurrence, column in pairs:
                        if table_occurrence in outer_table_occurrences:
                            outer_pairs.append((table_occurrence, column))
                        else:
                            inner_pairs.append((table_occurrence, column))
                    if len(outer_pairs) > 0 and len(inner_pairs) > 0:
                        join = Join(pairs)
                        new_joins.append(join)
            joins.extend(new_joins)
            joins = Join.merge_joins(joins)
            spj_query = SPJQuery(table_occurrences, joins, non_equi_join_predicates)
            return spj_query, [], equality_dict
        else:
            raise NotImplementedError("Unknown node type: " + pg_explain["Node Type"])
    elif "Scan" in pg_explain["Node Type"]:
        return _build_scan_query(schema, pg_explain)
    else:
        raise NotImplementedError("Unknown node type: " + pg_explain["Node Type"])
    return None


def _build_scan_query(schema: Schema, pg_explain: dict) -> Optional[Tuple[SPJQuery, List[str], Dict[float, List[Tuple[TableOccurrence, Column]]]]]:
    table_name = pg_explain["Relation Name"]
    table = schema.table(table_name)
    alias = pg_explain["Alias"]
    table_occurrence = TableOccurrence(table, alias=alias)
    if "Filter" in pg_explain:
        predicate = parse_table_predicate(table_occurrence, pg_explain["Filter"])
    else:
        predicate = TruePredicate()
    if isinstance(predicate, SimplePredicate):
        simple_predicates = [predicate]
    elif isinstance(predicate, Conjunction):
        simple_predicates = [predicate for predicate in predicate.predicates() if isinstance(predicate, SimplePredicate)]
    else:
        simple_predicates = []
    equality_dict = {}
    for simple_predicate in simple_predicates:
        if simple_predicate.operator() == COMPARISON_OPERATOR_EQ and isinstance(simple_predicate.value(), NumberLiteral):
            value = simple_predicate.value().value()
            if value not in equality_dict:
                equality_dict[value] = []
            equality_dict[value].append((table_occurrence, simple_predicate.column()))
    table_occurrence.set_predicate(predicate)
    if "Index Cond" in pg_explain:
        index_condition = pg_explain["Index Cond"]
    elif "Recheck Cond" in pg_explain:
        index_condition = pg_explain["Recheck Cond"]
    else:
        index_condition = None
    if index_condition is None:
        pushed_down_predicates = []
    else:
        join_alias, join_column, join_other_alias, join_other_column = _parse_equi_join_condition(index_condition)
        aliased_condition = f"({join_alias}.{join_column} = {join_other_alias}.{join_other_column})"
        pushed_down_predicates = [aliased_condition]
    spj_query = SPJQuery([table_occurrence], [], [])
    return spj_query, pushed_down_predicates, equality_dict


def _parse_equi_join_condition(condition: str) -> Tuple[str, str, str, str]:
    condition = re.sub(r"\((\w+\.\w+)\)::\w+(?: \w+)*(?:\(\d+(?:,\d+)?\))?", r"\1", condition)
    return re.match(r"\((.*)\.(.*) = (.*)\.(.*)\)", condition).groups()


def _duplicate_predicates_across_joins(table_occurrences: Iterable[TableOccurrence], joins: Iterable[Join]) -> None:
    column_to_join: Dict[Tuple[TableOccurrence, Column], Join] = {}
    for join in joins:
        for key in join.equivalence_class():
            column_to_join[key] = join

    additions: Dict[TableOccurrence, List[Predicate]] = {}
    for table_occurrence in table_occurrences:
        predicate = table_occurrence.predicate()
        if isinstance(predicate, Conjunction):
            candidates: List[Predicate] = list(predicate.predicates())
        else:
            candidates = [predicate]
        for candidate in candidates:
            if isinstance(candidate, TruePredicate):
                continue
            columns = candidate.columns_for(table_occurrence)
            if len(columns) != 1:
                continue
            column = next(iter(columns))
            key = (table_occurrence, column)
            join = column_to_join.get(key)
            if join is None:
                continue
            if isinstance(candidate, SimplePredicate) and candidate.operator() == COMPARISON_OPERATOR_EQ:
                join.set_redundant()
            if _predicate_has_column_cast(candidate):
                continue
            for other_to, other_col in join.equivalence_class():
                if (other_to, other_col) == key:
                    continue
                if other_col.data_type() != column.data_type():
                    continue
                new_predicate = _clone_predicate_to_column(candidate, table_occurrence, column, other_to, other_col)
                additions.setdefault(other_to, []).append(new_predicate)

    for table_occurrence, new_predicates in additions.items():
        existing = table_occurrence.predicate()
        if isinstance(existing, TruePredicate):
            existing_list: List[Predicate] = []
        elif isinstance(existing, Conjunction):
            existing_list = list(existing.predicates())
        else:
            existing_list = [existing]
        seen = {p.simple_hash() for p in existing_list}
        unique_new: List[Predicate] = []
        for p in new_predicates:
            h = p.simple_hash()
            if h in seen:
                continue
            seen.add(h)
            unique_new.append(p)
        if not unique_new:
            continue
        combined = existing_list + unique_new
        if len(combined) == 1:
            table_occurrence.set_predicate(combined[0])
        else:
            table_occurrence.set_predicate(Conjunction(combined))


def _predicate_has_column_cast(predicate: Predicate) -> bool:
    if isinstance(predicate, SimplePredicate):
        return predicate.column_cast_type() is not None
    if isinstance(predicate, Conjunction) or isinstance(predicate, Disjunction):
        return any(_predicate_has_column_cast(child) for child in predicate.predicates())
    if isinstance(predicate, Negation):
        return _predicate_has_column_cast(predicate.predicate())
    return False


def _clone_predicate_to_column(predicate: Predicate, src_to: TableOccurrence, src_col: Column,
                               dst_to: TableOccurrence, dst_col: Column) -> Predicate:
    if isinstance(predicate, SimplePredicate):
        assert predicate.table_occurrence() == src_to and predicate.column() == src_col
        return SimplePredicate(dst_to, dst_col, predicate.operator(), predicate.value(), predicate.column_cast_type())
    if isinstance(predicate, Conjunction):
        return Conjunction([_clone_predicate_to_column(child, src_to, src_col, dst_to, dst_col)
                            for child in predicate.predicates()])
    if isinstance(predicate, Disjunction):
        return Disjunction([_clone_predicate_to_column(child, src_to, src_col, dst_to, dst_col)
                            for child in predicate.predicates()])
    if isinstance(predicate, Negation):
        return Negation(_clone_predicate_to_column(predicate.predicate(), src_to, src_col, dst_to, dst_col))
    raise NotImplementedError("Cannot clone predicate of type: " + type(predicate).__name__)


def parse_table_predicate(table_occurrence: TableOccurrence, predicate_text: str) -> Predicate:
    sqlglot_expression = sqlglot.parse(predicate_text, dialect=Postgres)[0]
    return _sqlglot_parse(table_occurrence, sqlglot_expression)


_sqlglot_comparison_operator_map = {"eq": COMPARISON_OPERATOR_EQ,
                                    "neq": COMPARISON_OPERATOR_NEQ,
                                    "gt": COMPARISON_OPERATOR_GT,
                                    "gte": COMPARISON_OPERATOR_GTE,
                                    "lt": COMPARISON_OPERATOR_LT,
                                    "lte": COMPARISON_OPERATOR_LTE,
                                    "like": COMPARISON_OPERATOR_LIKE,
                                    "ilike": COMPARISON_OPERATOR_ILIKE,
                                    "is": COMPARISON_OPERATOR_IS,
                                    "regexplike": COMPARISON_OPERATOR_REGEX}


def _sqlglot_parse(table_occurrence: TableOccurrence, sqlglot_expression: sqlglot.Expression) -> Predicate:
    if sqlglot_expression.key == "paren":
        return _sqlglot_parse(table_occurrence, sqlglot_expression.args["this"])
    elif sqlglot_expression.key in _sqlglot_comparison_operator_map:
        this = sqlglot_expression.args["this"]
        expression = sqlglot_expression.args["expression"]
        this_is_column = _is_column_expression(this)
        expression_is_column = _is_column_expression(expression)
        if this_is_column and expression_is_column:
            comparison_operator = _sqlglot_comparison_operator_map[sqlglot_expression.key]
            left_column, left_cast_data_type = _sqlglot_parse_column(table_occurrence.table(), this)
            right_column, right_cast_data_type = _sqlglot_parse_column(table_occurrence.table(), expression)
            return ComparisonPredicate(table_occurrence, left_column, comparison_operator, right_column,
                                       left_column_cast_data_type=left_cast_data_type,
                                       right_column_cast_data_type=right_cast_data_type)
        if expression_is_column and not this_is_column:
            left = expression
            right = this
            reversed = True
        elif this_is_column and not expression_is_column:
            left = this
            right = expression
            reversed = False
        else:
            raise NotImplementedError("Comparison must be between column and literal: " + sqlglot_expression.key)
        if sqlglot_expression.key == "eq" and right.key == "any":
            inner_literals = _sqlglot_parse_array_literals(right)
            if len(inner_literals) == 1:
                comparison_operator = COMPARISON_OPERATOR_EQ
                literal = inner_literals[0]
            else:
                comparison_operator = COMPARISON_OPERATOR_IN
                literal = TupleLiteral(inner_literals)
        else:
            comparison_operator = _sqlglot_comparison_operator_map[sqlglot_expression.key]
            literal = _sqlglot_parse_literal(right)
        left_column, cast_data_type = _sqlglot_parse_column(table_occurrence.table(), left)
        if reversed:
            comparison_operator = comparison_operator.anti_symmetric_operator()
            assert comparison_operator is not None
        return SimplePredicate(table_occurrence, left_column, comparison_operator, literal, column_cast_data_type=cast_data_type)
    elif sqlglot_expression.key in ["and", "or"]:
        left = _sqlglot_parse(table_occurrence, sqlglot_expression.left)
        right = _sqlglot_parse(table_occurrence, sqlglot_expression.right)
        if sqlglot_expression.key == "and":
            predicates = []
            if isinstance(left, Conjunction):
                predicates.extend(left.predicates())
            else:
                predicates.append(left)
            if isinstance(right, Conjunction):
                predicates.extend(right.predicates())
            else:
                predicates.append(right)
            return Conjunction(predicates)
        elif sqlglot_expression.key == "or":
            predicates = []
            if isinstance(left, Disjunction):
                predicates.extend(left.predicates())
            else:
                predicates.append(left)
            if isinstance(right, Disjunction):
                predicates.extend(right.predicates())
            else:
                predicates.append(right)
            return Disjunction(predicates)
        else:
            raise NotImplementedError("Unknown SQLglot expression: " + sqlglot_expression.key)
    elif sqlglot_expression.key == "not":
        child = _sqlglot_parse(table_occurrence, sqlglot_expression.args["this"])
        return Negation(child)
    else:
        raise NotImplementedError("Unknown SQLglot expression: " + sqlglot_expression.key)

def _is_column_expression(sqlglot_expression: sqlglot.Expression) -> bool:
    if sqlglot_expression.key in ["cast", "paren"]:
        return _is_column_expression(sqlglot_expression.args["this"])
    elif sqlglot_expression.key == "column":
        return True
    else:
        return False

def _sqlglot_parse_column(table: Table, sqlglot_expression: sqlglot.Expression) -> Tuple[Column, DataType]:
    if sqlglot_expression.key == "cast":
        column, cast_data_type = _sqlglot_parse_column(table, sqlglot_expression.args["this"])
        assert cast_data_type is None
        if (isinstance(column.data_type(), NumericDataType) and sqlglot_expression.to.this in sqlglot_expression.to.STRING_TYPES) or (isinstance(column.data_type(), StringDataType) and sqlglot_expression.to.this in sqlglot_expression.to.NUMERIC_TYPES):
            data_type_string = sqlglot_expression.to.this.name.lower()
            cast_data_type = get_postgresql_data_type(data_type_string)
        else:
            cast_data_type = None
        return column, cast_data_type
    elif sqlglot_expression.key == "paren":
        return _sqlglot_parse_column(table, sqlglot_expression.args["this"])
    elif sqlglot_expression.key == "column":
        column_name = sqlglot_expression.name
        return table.column(column_name), None
    else:
        raise NotImplementedError("Unknown SQLglot expression: " + sqlglot_expression.key)


def _sqlglot_parse_literal(sqlglot_expression: sqlglot.Expression) -> Literal:
    if sqlglot_expression.key == "literal":
        if sqlglot_expression.is_string:
            return StringLiteral(sqlglot_expression.this.replace("%", "%%"))  # Escape % for LIKE
        elif sqlglot_expression.is_number:
            return NumberLiteral(Decimal(sqlglot_expression.this))
        else:
            raise NotImplementedError("Unknown SQLglot literal: " + sqlglot_expression.key)
    elif sqlglot_expression.key == "cast":
        literal = _sqlglot_parse_literal(sqlglot_expression.args["this"])
        if isinstance(literal, StringLiteral) and sqlglot_expression.to.this in sqlglot_expression.to.NUMERIC_TYPES:
            string_value = literal.value()
            return NumberLiteral(Decimal(string_value))
        else:
            return literal
    elif sqlglot_expression.key == "null":
        return NullLiteral()
    else:
        raise NotImplementedError("Unknown SQLglot expression: " + sqlglot_expression.key)

def _sqlglot_parse_array_literals(sqlglot_expression: sqlglot.Expression) -> List[Literal]:
    assert sqlglot_expression.key == "any"
    cast_expression = sqlglot_expression.args["this"]
    while cast_expression.key == "paren":
        cast_expression = cast_expression.args["this"]
    cast_type_expression = cast_expression.to
    assert cast_type_expression.this.name == "ARRAY"
    assert len(cast_type_expression.expressions) == 1
    array_type_expression = cast_type_expression.expressions[0]
    array_type = array_type_expression.this.name
    if array_type == "TEXT":
        array_string = cast_expression.this.this
        assert array_string[0] == "{"
        assert array_string[-1] == "}"
        array_string = array_string[1:-1]
        reader = csv.reader([array_string], skipinitialspace=True)
        values = next(reader)
        literals: List[Literal] = []
        for value in values:
            literals.append(StringLiteral(value))
        return literals
    if array_type == "INT":
        array_string = cast_expression.this.this
        assert array_string[0] == "{"
        assert array_string[-1] == "}"
        values = array_string[1:-1].split(",")
        literals = []
        for value in values:
            literals.append(NumberLiteral(Decimal(value)))
        return literals
    else:
        raise NotImplementedError("Array type not implemented: " + str(array_type))







