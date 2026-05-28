
from __future__ import annotations

import locale
from decimal import Decimal
from typing import Optional, Callable, Any
import re

locale.setlocale(locale.LC_COLLATE, 'en_US.UTF-8')

class ComparisonOperator:
    def __init__(self, name: str, symbol: str, evaluation_function: Optional[Callable[[Any, Any], Optional[bool]]]):
        self._name = name
        self._symbol = symbol
        self._evaluation_function = evaluation_function
        self._anti_symmetric_operator = None
        self._negated_operator = None

    def name(self) -> str:
        return self._name

    def symbol(self) -> str:
        return self._symbol

    def evaluate(self, left: Any, right: Any) -> bool:
        return self._evaluation_function(left, right)

    def set_anti_symmetric_operator(self, anti_symmetric_operator: ComparisonOperator) -> None:
        self._anti_symmetric_operator = anti_symmetric_operator

    def anti_symmetric_operator(self) -> Optional[ComparisonOperator]:
        return self._anti_symmetric_operator

    def set_negated_operator(self, negated_operator: ComparisonOperator) -> None:
        self._negated_operator = negated_operator

    def negated_operator(self) -> Optional[ComparisonOperator]:
        return self._negated_operator


def evaluate_like(value: str, pattern: str) -> bool:
    if value is None:
        return None

    pattern = pattern.replace("%%", "%")

    if "_" not in pattern:
        wild_card_count = pattern.count('%')
        # Pre-check: If no wildcards, use direct equality
        if wild_card_count == 0:
            return value == pattern
        elif wild_card_count == 1:
            # Handle single '%' patterns using fast string methods
            # 1. Starts with: 'abc%'
            if pattern.endswith('%'):
                return value.startswith(pattern[:-1])
            # 2. Ends with: '%abc'
            elif pattern.startswith('%'):
                return value.endswith(pattern[1:])
        elif wild_card_count == 2 and pattern.startswith('%') and pattern.endswith('%'):
            return pattern[1:-1] in value

    regex_p = re.escape(pattern)

    # 1. Normalize: Turn escaped '\%' back into '%' (for Python <3.7)
    regex_p = regex_p.replace(r'\%', '%').replace(r'\_', '_')

    # 2. Transform: Turn '%' into '.*' and '_' into '.'
    regex_p = regex_p.replace('%', '.*').replace('_', '.')

    regex = re.compile(f"^{regex_p}$", re.DOTALL)
    return regex.match(value) is not None


def evaluate_less(x: Any, y: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, (int, float, Decimal)) and isinstance(y, (int, float, Decimal)):
        return x < y
    if isinstance(x, str) and isinstance(y, str):
        return locale.strcoll(x, y) < 0
    raise ValueError("Cannot compare values: %s and %s" % (str(x), str(y)))


COMPARISON_OPERATOR_EQ = ComparisonOperator("equal", "=", lambda x, y: None if x is None else x == y)
COMPARISON_OPERATOR_NEQ = ComparisonOperator("not equal", "!=", lambda x, y: None if x is None else x != y)
COMPARISON_OPERATOR_LT = ComparisonOperator("less than", "<", evaluate_less)
COMPARISON_OPERATOR_GT = ComparisonOperator("greater than", ">", lambda x, y: None if x is None else evaluate_less(y, x))
COMPARISON_OPERATOR_LTE = ComparisonOperator("less than or equal", "<=", lambda x, y: None if x is None else not evaluate_less(y, x))
COMPARISON_OPERATOR_GTE = ComparisonOperator("greater than or equal", ">=", lambda x, y: None if x is None else not evaluate_less(x, y))
COMPARISON_OPERATOR_LIKE = ComparisonOperator("like", "LIKE", evaluate_like)
COMPARISON_OPERATOR_ILIKE = ComparisonOperator("ilike", "ILIKE", lambda x, y: None if x is None else evaluate_like(x.lower(), y.lower()))
COMPARISON_OPERATOR_IN = ComparisonOperator("in", "IN", lambda x, y: None if x is None else x in y)
COMPARISON_OPERATOR_IS = ComparisonOperator("is", "IS", lambda x, y: x is y)
COMPARISON_OPERATOR_REGEX = ComparisonOperator("regex", "~", lambda x, y: None if x is None else re.match(y, x) is not None)

COMPARISON_OPERATOR_EQ.set_anti_symmetric_operator(COMPARISON_OPERATOR_EQ)
COMPARISON_OPERATOR_NEQ.set_anti_symmetric_operator(COMPARISON_OPERATOR_NEQ)
COMPARISON_OPERATOR_LT.set_anti_symmetric_operator(COMPARISON_OPERATOR_GT)
COMPARISON_OPERATOR_GT.set_anti_symmetric_operator(COMPARISON_OPERATOR_LT)
COMPARISON_OPERATOR_LTE.set_anti_symmetric_operator(COMPARISON_OPERATOR_GTE)
COMPARISON_OPERATOR_GTE.set_anti_symmetric_operator(COMPARISON_OPERATOR_LTE)

COMPARISON_OPERATOR_EQ.set_negated_operator(COMPARISON_OPERATOR_NEQ)
COMPARISON_OPERATOR_NEQ.set_negated_operator(COMPARISON_OPERATOR_EQ)
COMPARISON_OPERATOR_LT.set_negated_operator(COMPARISON_OPERATOR_GTE)
COMPARISON_OPERATOR_GT.set_negated_operator(COMPARISON_OPERATOR_LTE)
COMPARISON_OPERATOR_LTE.set_negated_operator(COMPARISON_OPERATOR_GT)
COMPARISON_OPERATOR_GTE.set_negated_operator(COMPARISON_OPERATOR_LT)

COMPARISON_OPERATORS = {"=": COMPARISON_OPERATOR_EQ,
                        "!=": COMPARISON_OPERATOR_NEQ,
                        "<": COMPARISON_OPERATOR_LT,
                        ">": COMPARISON_OPERATOR_GT,
                        "<=": COMPARISON_OPERATOR_LTE,
                        ">=": COMPARISON_OPERATOR_GTE,
                        "LIKE": COMPARISON_OPERATOR_LIKE,
                        "ILIKE": COMPARISON_OPERATOR_ILIKE,
                        "IN": COMPARISON_OPERATOR_IN,
                        "IS": COMPARISON_OPERATOR_IS,
                        "~": COMPARISON_OPERATOR_REGEX}


