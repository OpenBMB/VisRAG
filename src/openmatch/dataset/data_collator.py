# Adapted from Tevatron (https://github.com/texttron/tevatron)

from dataclasses import dataclass
from transformers import DefaultDataCollator

def reshape(input_list):
    keys = input_list[0].keys()
    output_dict = {key: [] for key in keys}
    for data in input_list:
        for key in keys:
            output_dict[key].append(data[key]) # this is not append, it is extend

    return output_dict

def stack(list_of_lists):
    merged_list = []
    for sublist in list_of_lists:
        merged_list.extend(sublist)
    return merged_list

@dataclass
class MMQPCollator(DefaultDataCollator):

    def __call__(self, features):
        query = [f["query_"] for f in features]
        passages = [f["passages"] for f in features]
        
        # reshape
        query = reshape(stack(query)) # List[Dict[str, Any]] -> Dict[str, List[Any]]
        passages = reshape(stack(passages)) # List[Dict[str, Any]] -> Dict[str, List[Any]]
        
        return query, passages