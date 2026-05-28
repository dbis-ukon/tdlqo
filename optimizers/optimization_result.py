from relational_algebra_expressions.relational_algebra_expression import RelationalAlgebraExpression

class OptimizationResult():
    def __init__(self, plan: RelationalAlgebraExpression, optimization_time: None):
        self.plan = plan
        self.optimization_time = optimization_time
