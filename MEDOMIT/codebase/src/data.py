
import json
import os
import pickle
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from utils.data_objects import PatientCase, ACIPatientCase


def get_cases(enc, args, include_duplicate_users=False, included_encounters=None):

    return get_cases_aci(enc, args, include_duplicate_users, included_encounters)


def get_cases_aci(enc, args, include_duplicate_users, included_encounters):
    case_list = []
    for dataset_path, metadata_dataset_path in zip(args.dataset_path_list, args.metadata_dataset_path_list):
        with open(dataset_path) as json_file:
            dataset = json.load(json_file)

        metadata_df = pd.read_csv(metadata_dataset_path)

        for i, row_dict in enumerate(dataset["data"]):

            if metadata_df is not None:
                encounter_id, _ = row_dict["file"].split("-", 1)
                this_metadata = metadata_df[metadata_df["encounter_id"] == encounter_id.strip()]
                row_dict.update(this_metadata.iloc[0].to_dict())
            if "user_id" not in row_dict:
                row_dict["user_id"] = f"{dataset_path}:{i}"

            this_history = ACIPatientCase(enc, args, **row_dict)
            case_list.append(this_history)
    return case_list
