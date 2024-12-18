import asyncio
import json
import logging
import os
import re
from typing import Dict, List

from langchain import LLMChain, PromptTemplate
from langchain.llms.openai import OpenAI
from langchain.schema import BaseOutputParser
from tqdm import tqdm

from langchain_custom.sequential import CustomSequentialChain
from langchain.chat_models import ChatOpenAI
log = logging.getLogger()

class CategoryParser0(BaseOutputParser):

    def parse(self, output_list_of_dicts: List[Dict[str, str]]):
        for i in range(len(output_list_of_dicts)):
            for name in output_list_of_dicts[i].keys():
                if name == "medical_topic_groups":
                    output_list_of_dicts[i][name] = re.sub(
                        r"--omittedMedTopic\s\(start\)\s--.*--omittedMedTopic\s\(end\)\s--", "",
                        output_list_of_dicts[i][name], flags=re.DOTALL | re.MULTILINE).strip()

        return output_list_of_dicts


class OmissionParser0(BaseOutputParser):

    def parse(self, output_list_of_dicts: List[Dict[str, str]]):
        for i in range(len(output_list_of_dicts)):
            for name in output_list_of_dicts[i].keys():
                if name == "subjective_and_omissions":
                    output_list_of_dicts[i]["corrupted_subjective"], output_list_of_dicts[i]["omitted_facts"] = \
                    output_list_of_dicts[i][name].split("Omitted Facts:", 1)
        return output_list_of_dicts


async def async_generate(seq_chain, input_dict):
    is_chronic_or_refill, llm_output = await seq_chain.agenerate(input_dict)
    return is_chronic_or_refill, llm_output


async def generate_concurrently(seq_chain, input_list):
    tasks = [async_generate(seq_chain, input_dict) for input_dict in input_list]
    result_list = await asyncio.gather(*tasks)
    return result_list


def batch_chain(chain, input_list, extract_input_fn, extract_output_fn, batch_size=2, **kwargs):
    chain_name = None
    if "chain_name" in kwargs:
        chain_name = kwargs["chain_name"]
    pbar = tqdm(total=len(input_list), desc=chain_name)

    for batch_start in range(0, len(input_list), batch_size):
        batch_list = []
        for batch_item in input_list[batch_start:batch_start + batch_size]:
            input_dict = extract_input_fn(batch_item, **kwargs)
            batch_list.append(input_dict)
            for chain_indx, individual_chain in enumerate(chain.chains):
                prompt_text = individual_chain.prompt.format_prompt(**input_dict).text
                if type(batch_item) is not dict:
                    batch_item.prompts[f"{chain_name}_{chain_indx}"] = prompt_text

        loop = asyncio.get_event_loop()
        coroutine = generate_concurrently(chain, batch_list)
        results = loop.run_until_complete(coroutine)
        for i in range(len(results)):
            results_value = results[i][0]
            input_list[batch_start + i] = extract_output_fn(input_list[batch_start + i], results_value, **kwargs)
            pbar.update()
    return input_list


def configure_template(template_config, summarization_template_dir):
    output_parser = None
    output_parser_dict = {
        "CategoryParser0": CategoryParser0(),
        "OmissionParser0" : OmissionParser0(),
    }

    if "output_parser" in template_config and template_config["output_parser"] in output_parser_dict:
        output_parser = output_parser_dict[template_config["output_parser"]]

    with open(os.path.join(summarization_template_dir, template_config[f"template_path"]), 'r') as template_file:
        template = template_file.read()
    prompt_template = PromptTemplate(
        input_variables=template_config[f"input_variables"],
        template=template,
        output_parser=output_parser
    )
    return prompt_template


def setup_chains(args, default_llm, directory_name):
    seq_chain_dict = {}
    summarization_template_dir = os.path.join(args.template_directory, directory_name)
    for json_file in [x for x in os.listdir(summarization_template_dir) if x.endswith(".json")]:
        template_name_split = json_file.split(".")
        template_name = template_name_split[0]

        template_list = []
        input_vars = set()
        output_vars = set()
        with open(os.path.join(summarization_template_dir, json_file), 'r') as json_in:
            json_config = json.load(json_in)
        for template_number, template_config in enumerate(json_config):
            if "llm_config" in template_config:
                default_llm_args = {k: v for k, v in default_llm._default_params.items() if
                                    k not in ["model", "model_name"]}
                if args.ignore_stream:
                    default_llm_args.update(
                        {k: v for k, v in template_config["llm_config"].items() if k not in ["model", "model_name", "stream"]})
                else:
                    default_llm_args.update(
                        {k: v for k, v in template_config["llm_config"].items() if k not in ["model", "model_name"]})
                model_name = default_llm._default_params["model"]
                if "model_name" in template_config["llm_config"]:
                    model_name = template_config["llm_config"]["model_name"]
                    log.info(f"template number {template_number}, name{template_name}, model {model_name}")
                if model_name.startswith("gpt-3.5-turbo") or model_name.startswith("gpt-4"):
                    llm = ChatOpenAI(openai_api_key=default_llm.openai_api_key, model_name=model_name,
                                 **default_llm_args)
                else:
                    llm = OpenAI(openai_api_key=default_llm.openai_api_key, model_name=model_name,
                                 **default_llm_args)
            else:
                llm = default_llm
            prompt_template = configure_template(template_config, summarization_template_dir)

            chain = LLMChain(llm=llm, prompt=prompt_template,
                             output_key=template_config["output_key"])
            template_list.append(chain)
            input_vars.update(template_config["input_variables"])
            output_vars.add(template_config["output_key"])
        overall_chain = CustomSequentialChain(
            chains=template_list,
            input_variables=list(set(input_vars) - set(output_vars)),
            output_variables=list(output_vars),
            verbose=False)
        seq_chain_dict[template_name] = overall_chain
    return seq_chain_dict
