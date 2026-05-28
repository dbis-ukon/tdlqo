from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression

from relational_algebra_expressions.join_expressions.join_expression import JoinExpression
from relational_algebra_expressions.join_expressions.hash_join_expression import HashJoinExpression
from relational_algebra_expressions.join_expressions.merge_join_expression import MergeJoinExpression
from relational_algebra_expressions.join_expressions.nested_loop_join_expression import NestedLoopJoinExpression

from relational_algebra_expressions.scan_expressions.scan_expression import ScanExpression
from relational_algebra_expressions.scan_expressions.bitmap_index_scan_expression import BitmapIndexScanExpression
from relational_algebra_expressions.scan_expressions.index_scan_expression import IndexScanExpression
from relational_algebra_expressions.scan_expressions.index_only_scan_expression import IndexOnlyScanExpression
from relational_algebra_expressions.scan_expressions.tid_scan_expression import TidScanExpression
from relational_algebra_expressions.scan_expressions.sequential_scan_expression import SequentialScanExpression

from queries.table_occurrence import TableOccurrence
from optimizers.optimization_result import OptimizationResult
from queries.spj_query import SPJQuery
from schemas.schema import Schema


def hint_str(plan):
    from execution_engines.pg_hint_plan_execution_engine import PgHintPlanExecutionEngine 
    hint_str  = "/*+ %s */" % " ".join(PgHintPlanExecutionEngine._build_hints(plan))
    return hint_str

# parses a explain json to a rax plan
def parse_json_to_optimization_result(result, query: SPJQuery , schema: Schema) -> OptimizationResult:
        planning_time = None
        if 'Planning Time' in result[0]: #None if result is 
            planning_time = result[0]['Planning Time']
        
        result = result[0]['Plan']

        def _extract(node):
            nodetype = node['Node Type']
            
            if nodetype in ["Aggregate", "Gather", "Hash", "Materialize", "Sort", "Memoize", "Gather Merge"]:
                assert len(node['Plans']) == 1
                return _extract(node['Plans'][0])
            
            elif nodetype in ['Nested Loop', 'Merge Join', 'Hash Join']:
                children = node['Plans']
                assert len(children) == 2
                left = _extract(children[0])
                right = _extract(children[1])
                join_condition = None

                if nodetype == 'Nested Loop':
                    return NestedLoopJoinExpression(left, right, join_condition)
                elif nodetype == 'Merge Join':
                    return MergeJoinExpression(left, right, join_condition)
                elif nodetype == 'Hash Join':
                    return HashJoinExpression(left, right, join_condition)
            
            elif nodetype in ['Index Scan', 'Index Only Scan', 'Seq Scan', 'Tid Scan', 'Bitmap Heap Scan']:
                table_name = node['Relation Name']
                table = schema.table(table_name)
                table_occurrence = query.table_occurrence_by_alias(node['Alias'])

                if nodetype == 'Seq Scan':
                    return SequentialScanExpression(table_occurrence)
                elif nodetype == 'Tid Scan':
                    return TidScanExpression(table_occurrence)

                elif nodetype == 'Bitmap Heap Scan':
                    #child_node = node['Plans'][0]
                    #assert child_node['Node Type'] == 'Bitmap Index Scan'
                    #index_name = child_node['Index Name']
                    #index = table.index(index_name)
                    return BitmapIndexScanExpression(table_occurrence, None)
                
                index_name = node['Index Name']
                try:
                    index = table.index(index_name)
                except:
                    index = None
                if nodetype == 'Index Scan':
                    return IndexScanExpression(table_occurrence, index)
                
                elif nodetype == 'Index Only Scan':
                    return IndexOnlyScanExpression(table_occurrence, index)

            else:
                raise NotImplementedError
        
        parsed_result = _extract(result)
        return OptimizationResult(parsed_result, planning_time)