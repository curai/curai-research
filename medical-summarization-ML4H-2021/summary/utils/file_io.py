"""General File I/O Utilities"""
import json
from pathlib import Path

def read_json(json_file): 
    with open(json_file, 'r') as f: 
        data = json.load(f)
    return data

def write_json(data, out_file): 
    out_file = Path(out_file)
    dir_name = out_file.parents[0] 
    dir_name.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f: 
        json.dump(data, f, indent=4)

