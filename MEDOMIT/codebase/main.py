import logging
import os

import openai
import tiktoken
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.globals import set_llm_cache

from modules import simulate_cases
from src.data import get_cases
from utils.init_utils import parse_args, setup_logging

openai.log = None
openai.util.logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.WARNING)
log = logging.getLogger()


def main():
    os.makedirs("cache", exist_ok=True)
    set_llm_cache(SQLiteCache(database_path="cache/langchain.db"))
    args = parse_args()
    log = setup_logging(logging.getLogger(), args)
    default_llm = ChatOpenAI(model_name=args.model_name,
                             temperature=args.temperature,
                             max_tokens=args.max_tokens,
                             openai_api_key=args.openai_api_key)
    enc = tiktoken.encoding_for_model(args.model_name)
    if args.mother_encounter_id_file is not None:
        encounters_to_include = [x.strip() for x in open(args.mother_encounter_id_file).readlines()]
    else:
        encounters_to_include = set()
    case_list = get_cases(enc, args, included_encounters=encounters_to_include)

    if args.mother_encounter_id_file is not None:
        included_cases = [case for case in case_list if case.mother_encounter_id in encounters_to_include]
        other_cases = [case for case in case_list if case.mother_encounter_id not in encounters_to_include][
                      :args.example_limit]
        case_list = included_cases + other_cases
    elif args.example_limit > 0:
        case_list = case_list[:args.example_limit]
    simulate_cases(case_list, args, default_llm)





if __name__ == '__main__':
    main()
