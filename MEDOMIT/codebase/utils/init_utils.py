
import time
import datetime

import configargparse
import os
import logging

import openai


def setup_logging(log, args):
    log.handlers.clear()
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(os.path.join(args.output_path, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.setLevel(logging.INFO)
    log.addHandler(ch)
    log.info(args)
    log.info(f"Writing to {args.output_path}")
    return log


def parse_args():
    p = configargparse.ArgParser(default_config_files=['default.conf'])
    p.add('-c', '--my-config', is_config_file=True, help='config file path')
    p.add('--debug', action='store_true')
    p.add_argument("--top_p", type=float, default=1.)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=250)
    p.add('--openai_api_key', required=True)
    p.add('--template_directory', type=str)
    p.add('--history_type', type=str)
    p.add_argument("--max_history_length", type=int, default=4_750)
    p.add('--ignore_stream', action='store_true')
    p.add('--model_list',  nargs='*')
    p.add('--dataset', default="aci")
    p.add('--dataset_path_list', nargs='+', default=["data/aci/challenge_data_json/train_subjective_truncated.json"])
    p.add('--metadata_dataset_path_list', nargs='+', default=["data/aci/challenge_data/train_metadata.csv"])

    p.add('--min_dialogue_lines', default=10)
    p.add('--model_name', default="gpt-4")

    p.add('--timestamp', default=None)
    p.add('--source_query', default="subjective")

    p.add('--example_limit', default=10, type=int)

    p.add('--mother_encounter_id_file', default=None, type=str)

    p.add('--ddx_completion_score', action='store_true')

    p.add('--llm_eoc_categorization', action='store_true')
    p.add('--regenerate_output', action='store_true')
    p.add(
        '--new_note_start_date',
        type=datetime.date.fromisoformat,
        default=datetime.date(2023, 4, 5),
    )
    args = p.parse_args()
    os.environ['OPENAI_API_KEY'] = args.openai_api_key
    openai.api_key = args.openai_api_key
    if args.timestamp is None:
        args.timestamp = time.strftime('%Y%m%d_%H%M%S')
    if args.debug:
        args.output_path = os.path.join("debug", args.timestamp)
    else:
        args.output_path = os.path.join("output", args.timestamp)
    os.makedirs(args.output_path, exist_ok=True)
    return args
