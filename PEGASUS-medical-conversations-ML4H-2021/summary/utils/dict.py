"""General Dictionary Utilities"""
class AttrDict(dict):
    """Dictionary subclass whose entries can be accessed by attributes
    (as well as normally)"""
    def __init__(self, *args, **kwargs):
        def from_nested_dict(data):
            """Construct nested AttrDicts from nested dictionaries"""
            if not isinstance(data, dict):
                return data
            else:
                return AttrDict({key: from_nested_dict(data[key])
                                    for key in data})

        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

        for key in self.keys():
            self[key] = from_nested_dict(self[key])      


def intersect_freq_dicts(dict1, dict2): 
    """Take the intersection of two frequency
    dictionaries"""
    intersect_dict = {}
    all_keys = list(dict1.keys()) + list(dict2.keys())
    for key in all_keys: 
        if key not in dict1 or key not in dict2: 
            continue 
        intersect_dict[key] = min(dict1[key], dict2[key])
    return intersect_dict

def subset_dict(all_dict, keys_subset): 
    """Take a subset of dict given keys"""
    subset = {k: all_dict[k] for k in keys_subset}
    return subset
