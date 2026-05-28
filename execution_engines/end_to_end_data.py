from typing import Optional

from execution_engines.execution_data import ExecutionData


class EndToEndData:
    def __init__(self,
                 optimizer_id: Optional[str],
                 execution_data: ExecutionData,
                 optimization_time: Optional[float],
                 memorization_time: Optional[float]):
        self.optimizer_id = optimizer_id
        self.execution_data = execution_data
        self.optimization_time = optimization_time
        self.memorization_time = memorization_time

    def to_string(self) -> str:
        if self.optimization_time is None:
            optimization_time_str = "N/A"
        else:
            optimization_time_str = f"{self.optimization_time:.2f} ms"
        if self.memorization_time is None:
            memorization_time_str = "N/A"
        else:
            memorization_time_str = f"{self.memorization_time:.2f} ms"
        return "%s, %s, %s" % (self.execution_data.to_string(), optimization_time_str, memorization_time_str)

    def to_csv(self) -> str:
        if self.optimizer_id is None:
            optimizer_id_str = "-"
        else:
            optimizer_id_str = self.optimizer_id
        if self.optimization_time is None:
            optimization_time_str = "-"
        else:
            optimization_time_str = f"{self.optimization_time}"
        if self.memorization_time is None:
            memorization_time_str = "-"
        else:
            memorization_time_str = f"{self.memorization_time}"
        return "%s,%s,%s,%s" % (optimizer_id_str, self.execution_data.to_csv(), optimization_time_str, memorization_time_str)









