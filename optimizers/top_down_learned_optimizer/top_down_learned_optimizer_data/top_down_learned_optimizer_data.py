import torch
import torch_geometric

class TopDownLearnedOptimizerData(torch_geometric.data.Data):
    def __init__(self,
                 table_occurrence_information: torch.FloatTensor,
                 table_index: torch.LongTensor,
                 table_specific_information: torch.FloatTensor,
                 join_information: torch.FloatTensor,
                 join_information_index: torch.LongTensor,
                 join_index: torch.LongTensor,
                 join_indptr: torch.LongTensor,
                 join_count: torch.FloatTensor,
                 join_operator_batch: torch.LongTensor,
                 join_operator_information: torch.FloatTensor,
                 join_operator_subgroup_table_index: torch.LongTensor,
                 join_operator_subgroup_table_information: torch.FloatTensor,
                 join_operator_subgroup_index: torch.LongTensor,
                 join_operator_subgroup_indptr: torch.LongTensor,
                 join_operator_subgroup_count: torch.FloatTensor,
                 requirement_information: torch.FloatTensor,
                 join_operator_outer_subgroup_index: torch.LongTensor,
                 join_operator_inner_subgroup_index: torch.LongTensor,
                 join_operator_index: torch.LongTensor,
                 join_operator_join_index: torch.LongTensor,
                 join_operator_join_indptr: torch.LongTensor,
                 join_operator_join_count: torch.FloatTensor,
                 join_operator_num_tables: torch.FloatTensor,
                 scan_operator_batch: torch.LongTensor,
                 scan_operator_information: torch.FloatTensor,
                 scan_operator_index: torch.LongTensor,
                 batch_indptr: torch.LongTensor,
                 batch_count: torch.FloatTensor
                 ):
        super().__init__(x=table_occurrence_information, edge_index=join_index)
        self.table_index = table_index
        self.table_specific_information = table_specific_information
        self.join_information = join_information
        self.join_information_index = join_information_index
        self.join_indptr = join_indptr
        self.join_count = join_count
        self.join_operator_batch = join_operator_batch
        self.join_operator_information = join_operator_information
        self.join_operator_subgroup_table_index = join_operator_subgroup_table_index
        self.join_operator_subgroup_table_information = join_operator_subgroup_table_information
        self.join_operator_subgroup_index = join_operator_subgroup_index
        self.join_operator_subgroup_indptr = join_operator_subgroup_indptr
        self.join_operator_subgroup_count = join_operator_subgroup_count
        self.requirement_information = requirement_information
        self.join_operator_outer_subgroup_index = join_operator_outer_subgroup_index
        self.join_operator_inner_subgroup_index = join_operator_inner_subgroup_index
        self.join_operator_index = join_operator_index
        self.join_operator_join_index = join_operator_join_index
        self.join_operator_join_indptr = join_operator_join_indptr
        self.join_operator_join_count = join_operator_join_count
        self.join_operator_num_tables = join_operator_num_tables
        self.scan_operator_batch = scan_operator_batch
        self.scan_operator_information = scan_operator_information
        self.scan_operator_index = scan_operator_index
        self.batch_indptr = batch_indptr
        self.batch_count = batch_count

    def __inc__(self, key, value, *args, **kwargs):
        if key in ["join_information_index"]:
            return self.edge_index.size(1)
        elif key in ["join_operator_index"]:
            return self.join_operator_information.size(0)
        elif key in ["join_operator_subgroup_table_index",
                     "scan_operator_index",
                     "batch_indptr"]:
            return self.x.size(0)
        elif key in ["join_operator_subgroup_index",
                     "join_operator_outer_subgroup_index",
                     "join_operator_inner_subgroup_index"]:
            if self.join_operator_subgroup_index.numel() == 0:
                return 0
            return self.join_operator_subgroup_index.max().item() + 1
        elif key in ["join_operator_join_index",
                     "join_indptr"]:
            return self.edge_index.size(1)
        elif key in ["join_operator_subgroup_indptr"]:
            return self.join_operator_subgroup_index.size(0)
        elif key in ["join_operator_join_indptr"]:
            return self.join_operator_join_index.size(0)
        elif key in ["join_operator_batch", "scan_operator_batch"]:
            return 1
        elif key in ["table_index", "batch_count"]:
            return 0
        else:
            return super().__inc__(key, value)

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        return super().__cat_dim__(key, value, *args, **kwargs)





