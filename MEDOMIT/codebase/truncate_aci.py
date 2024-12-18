
import json
import logging
import os
import time
from collections import defaultdict

import langchain
import pandas as pd
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from openai.error import RateLimitError
from tqdm import tqdm
from langchain.globals import set_llm_cache

from langchain_custom.langchain_utils import setup_chains, batch_chain
from utils.init_utils import parse_args, setup_logging


def main():
    os.makedirs("cache", exist_ok=True)
    set_llm_cache(SQLiteCache(database_path="cache/langchain.db"))
    args = parse_args()
    log = setup_logging(logging.getLogger(), args)
    default_llm = ChatOpenAI(model_name=args.model_name,
                             temperature=args.temperature,
                             max_tokens=args.max_tokens,
                             openai_api_key=args.openai_api_key)
    dataset_path_list = [#"data/aci/challenge_data_json/clef_taskC_test3_subjective.json",
                         "data/aci/challenge_data_json/clinicalnlp_taskB_test1_subjective.json"
                         #"data/aci/challenge_data_json/clinicalnlp_taskC_test2_subjective.json"
                         ]
    for dataset_path in dataset_path_list:
        truncate_dataset(args, dataset_path, default_llm, log)


def truncate_dataset(args, dataset_path, default_llm, log):
    with open(dataset_path) as json_file:
        list_of_dicts = json.load(json_file)["data"]
    output_dataset_path = dataset_path.replace(".json", "_truncated.json")
    truncate_dialogue_chain = setup_chains(args, default_llm, "truncate_dialogue")

    def truncate_dialogue_input_fn(batch_item, **kwargs):
        dialogue_with_lines = ""
        for i, chat_line in enumerate(batch_item['src'].split("\n")):
            dialogue_with_lines += f"{i}| {chat_line}\n"
        return {"dialogue": dialogue_with_lines}

    def truncate_dialogue_output_fn(batch_item, result, **kwargs):
        if result["first_line_to_remove"].lower().strip() == "none":
            batch_item['src'] = ''
        else:
            result_dict = json.loads(result["first_line_to_remove"])
            for i, line_num in enumerate(result_dict):
                if i > 0:
                    log.info(result_dict)
                    break
                dialogue_to_include = "\n".join(batch_item['src'].split("\n")[:int(line_num) + 1])
                batch_item['src'] = dialogue_to_include

        return batch_item

    for chain_name, chain in truncate_dialogue_chain.items():
        new_list_of_dicts = []
        pbar = tqdm(total=len(list_of_dicts), desc=chain_name)
        stats = defaultdict(int)
        for row_dict in list_of_dicts:
            chain_input = truncate_dialogue_input_fn(row_dict)
            result_returned = False
            while result_returned is False:
                try:
                    result = chain(chain_input)
                    result_returned = True
                except RateLimitError as e:
                    time.sleep(60)
                    stats["retries"] += 1

            output_row_dict = truncate_dialogue_output_fn(row_dict, result)
            new_list_of_dicts.append(output_row_dict)
            pbar.update()
            pbar.set_postfix(stats)

    new_list_of_dicts = []
    for case in list_of_dicts:
        dialogue_lines = len(case['src'].split("\n"))
        if dialogue_lines > args.min_dialogue_lines:
            new_list_of_dicts.append(case)
        else:
            log.info(f"Skipped {case}")
    output = {"data": list_of_dicts}
    with open(output_dataset_path, 'w') as out_f:
        json.dump(output, out_f)
    output_df = pd.DataFrame.from_dict(new_list_of_dicts)
    output_df.to_csv(dataset_path.replace(".json", ".csv"), index=False)


if __name__ == "__main__":
    main()
