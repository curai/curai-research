
import json
import logging
import time
from collections import defaultdict
import vertexai
from langchain_google_vertexai import VertexAI


from tqdm import tqdm

from langchain_custom.langchain_utils import setup_chains, batch_chain, configure_template
from src.output import output_results
from utils.metrics import calculate_metrics

log = logging.getLogger()

def simulate_cases(case_list, args, default_llm):
    if "ext" not in args.dataset:
        case_list = generate_summary_vertex(case_list, args, default_llm)
    case_list = extract_facts_truncated(case_list, args, default_llm)
    case_list = generate_ddx_original(case_list, args, default_llm)
    #case_list = ddx_delta(case_list, args)

    case_list = fact_clustering(case_list, args, default_llm)
    case_list = fact_sub_clustering(case_list, args, default_llm)

    case_list = fact_relevance(case_list, args, default_llm)
    case_list = find_omissions_baseline(case_list, args, default_llm)
    case_list = calculate_metrics(case_list, args, default_llm)
    output_results(args, case_list, log)



def extract_facts_truncated(case_list, args, default_llm):
    extract_fact_chain = setup_chains(args, default_llm, "extract_facts")

    def extract_facts_input_fn(batch_item, **kwargs):

        return {"dialogue": batch_item.dialogue}

    def extract_facts_output_fn(batch_item, result, **kwargs):
        fact_json = result["facts"]
        facts = json.loads(fact_json)
        fact_set = set()
        for fact_type, fact_list in facts.items():
            fact_list_with_ids = [(f"F{i}", fact) for i, fact in
                                  zip(range(len(fact_set), len(fact_set) + len(fact_list)), fact_list) if
                                  fact not in fact_set]
            fact_set.update(fact_list)
            batch_item.facts[fact_type] = fact_list_with_ids
        return batch_item

    for chain_name, chain in extract_fact_chain.items():
        case_list = batch_chain(chain, case_list, extract_facts_input_fn, extract_facts_output_fn,
                                chain_name=chain_name)
    return case_list
def generate_summary_vertex(case_list, args, default_llm):
    vertexai.init()

    vertex_llm = VertexAI(model_name="medlm-large",max_output_tokens=500, cache=False)
    with open("templates/summary/summary.json", 'r') as json_in:
        json_config = json.load(json_in)
    template_config = json_config[0]
    prompt_template = configure_template(template_config, "templates/summary")
    chain = prompt_template | vertex_llm
    def summary_input_fn(batch_item, **kwargs):
        return {"dialogue": batch_item.dialogue}

    def summary_output_fn(batch_item, result, **kwargs):
        batch_item.subjective_output = result.get("subjective", "")

        return batch_item
    pbar = tqdm(total=len(case_list))
    for i in range(len(case_list)):
        input_dict = summary_input_fn(case_list[i])
        prompt = prompt_template.format_prompt(**input_dict)
        case_list[i].subjective_output = vertex_llm(prompt.to_string())
        pbar.update()
        time.sleep(20)
    return case_list

def generate_summary(case_list, args, default_llm):
    summary_chains = setup_chains(args, default_llm, "summary")

    def summary_input_fn(batch_item, **kwargs):
        return {"dialogue": batch_item.dialogue}

    def summary_output_fn(batch_item, result, **kwargs):
        batch_item.subjective_output = result.get("subjective", "")

        return batch_item

    for chain_name, chain in summary_chains.items():
        case_list = batch_chain(chain, case_list,
                                summary_input_fn,
                                summary_output_fn,
                                chain_name=chain_name)
    return case_list


def fact_clustering(case_list, args, default_llm):
    clustering_chains = setup_chains(args, default_llm, "clustering")

    def clustering_input_fn(batch_item, **kwargs):

        return {"ddx": batch_item.get_ddx_string(kwargs['ddx_chain_name']),
                "facts": batch_item.get_fact_list_str()}

    def clustering_output_fn(batch_item, result, **kwargs):
        fact_cluster_blocks = json.loads(result["fact_clusters"])

        batch_item.fact_clusters[f"{kwargs['chain_name']}_{kwargs['ddx_chain_name']}"] = fact_cluster_blocks

        return batch_item

    for ddx_chain_name in case_list[0].ddx_dialogue.keys():
        for chain_name, chain in clustering_chains.items():
            case_list = batch_chain(chain, case_list, clustering_input_fn, clustering_output_fn,
                                    chain_name=chain_name, ddx_chain_name=ddx_chain_name)
    return case_list


def fact_sub_clustering(case_list, args, default_llm):
    clustering_chains = setup_chains(args, default_llm, "subclustering")

    def clustering_input_fn(batch_item, **kwargs):

        return {"ddx": batch_item.get_ddx_string(kwargs['ddx_chain_name']),
                "facts": batch_item.get_fact_list_str()}

    def clustering_output_fn(batch_item, result, **kwargs):
        fact_cluster_blocks = result["fact_clusters"].split("---")

        batch_item.fact_clusters[kwargs['ddx_chain_name']] = {}
        for block in fact_cluster_blocks:
            block_lines = block.strip().split("\n")
            _, block_info = block_lines[0].split(":")
            batch_item.fact_clusters[f"{kwargs['ddx_chain_name']}_{kwargs['chain_name']}"][block_info] = [x.strip() for
                                                                                                          x in
                                                                                                          block_lines[
                                                                                                          1:]]
        return batch_item

    pbar = tqdm()
    stats = defaultdict(lambda: 0)
    for fact_cluster_config in case_list[0].fact_clusters.keys():

        for chain_name, chain in clustering_chains.items():
            if ("negative" in fact_cluster_config and "negative" in chain_name) or ("negative" not in fact_cluster_config and "negative" not in chain_name):
                for case in case_list:
                    case.fact_subclusters[f"{chain_name}_{fact_cluster_config}"] = {}
                    for fact_cluster_name, fact_cluster in case.fact_clusters[fact_cluster_config].items():
                        if len(fact_cluster) == 0 :
                            continue
                        try:
                            input_dict = {"diagnosis": fact_cluster_name,
                                          "facts": "\n".join(f"{x}:{v}" for x, v in fact_cluster.items())}
                            for chain_indx, individual_chain in enumerate(chain.chains):
                                prompt_text = individual_chain.prompt.format_prompt(**input_dict).text
                                case.prompts[f"{chain_name}_{chain_indx}"] = prompt_text
                            results = chain(input_dict)['fact_sub_clusters']
                            case.fact_subclusters[f"{chain_name}_{fact_cluster_config}"][fact_cluster_name] = json.loads(
                                results)
                        except Exception as e:
                            case.fact_subclusters[f"{chain_name}_{fact_cluster_config}"][fact_cluster_name] = results
                            stats["parsing_error"] += 1
                            print(e)
                        pbar.set_postfix(stats)
                        pbar.update()

    return case_list






"""
Required dependencies: (generate_ddx_original and/or generate_ddx_corruption) and extract_facts

Takes the output of extract_facts, which is a list of facts, and a ddx, and outputs the importance of each fact.
What form the importance takes (e.g. score, ranking, etc.) is variable.
"""


def fact_relevance(case_list, args, default_llm):
    fact_chains = setup_chains(args, default_llm, "fact_relevance")

    def relevance_input_fn(batch_item, **kwargs):

        return {"ddx": batch_item.get_ddx_string(kwargs["ddx_chain_name"]), "facts": batch_item.get_fact_list_str()}

    def relevance_output_fn(batch_item, result, **kwargs):
        # the expected format is (fact number id, fact categorization/score/ranking, fact)
        batch_item.facts_relevance_output[kwargs["chain_name"]] = result.get("output_str") or result.get("fact_ranking",
                                                                                                         "")

        if result.get("output_str"):
            batch_item.facts_relevance[kwargs["chain_name"]] = {}
            current_category = None
            for line in result.get("output_str").split("\n"):
                if line.startswith("Category|"):
                    _, current_category = line.split("|")
                elif len(line.strip()) > 0:
                    fact_rank, fact_id, fact = line.split("|")
                    batch_item.facts_relevance[kwargs["chain_name"]][fact_id] = {"rank": fact_rank,
                                                                                 "category": current_category,
                                                                                 "fact": fact}



        elif result.get("fact_ranking"):
            batch_item.facts_relevance[kwargs["chain_name"]] = []

            for fact_str in result["fact_ranking"].split("\n"):
                if len(fact_str.strip()) > 0:
                    ranking, fact_id, fact = fact_str.split(":", 2)
                    batch_item.facts_relevance[kwargs["chain_name"]].append(
                        (fact_id.strip(), ranking.strip(), fact.strip()))
        return batch_item

    for ddx_chain_name in case_list[0].ddx_dialogue.keys():
        for chain_name, chain in fact_chains.items():
            case_list = batch_chain(chain, case_list,
                                    relevance_input_fn,
                                    relevance_output_fn,
                                    chain_name=chain_name,
                                    ddx_chain_name=ddx_chain_name)
    return case_list


"""
Function dependencies: extract_facts

This function takes in an actual subjective and a list of facts, and outputs a corrupted subjective with a list of the omitted facts.
"""



"""
Required dependencies: corrupt_subjectives

Given a (corrupted) subjective and the subjective input (chat, etc.), find omitted facts.
"""




def find_omissions_baseline(case_list, args, default_llm):
    identify_omissions = setup_chains(args, default_llm, "identify_omissions")

    def omissions_input_fn(batch_item, **kwargs):
        return {"subjective": batch_item.subjective_output,
                "fact_list": batch_item.get_fact_list_str()}

    def omissions_output_fn(batch_item, result, **kwargs):
        lettered_missing_fact_list = json.loads(result["missing_facts"])

        batch_item.omissions_found_baseline[f"{kwargs['chain_name']}"] = lettered_missing_fact_list
        return batch_item

    for chain_name, chain in identify_omissions.items():
        case_list = batch_chain(chain, case_list, omissions_input_fn, omissions_output_fn, chain_name=chain_name)
    return case_list


"""
Required dependencies: find_omissions, fact_relevance

Match the facts from find_omissions to those in fact_relevance
This step is required because the two fact lists (from the corruption and from the ground truth) are generated separately
Therefore we can score the importance of each corrupted fact
"""

"""
Generates a ddx from the original sujective.
"""


def generate_ddx_original(case_list, args, default_llm):
    ddx_chains = setup_chains(args, default_llm, "ddx")

    def ddx_input_fn(batch_item, **kwargs):
        return {"subjective": batch_item.subjective_output}

    def ddx_output_fn(batch_item, result, **kwargs):
        ddx_list = [x.split("|") for x in result['ddx'].split('\n') if len(x.strip()) > 0]
        if len(ddx_list) == 1 and ddx_list[0][0].upper().strip() == "NONE":
            batch_item.ddx[kwargs["chain_name"]] = []

        else:
            batch_item.ddx[kwargs["chain_name"]] = ddx_list
        return batch_item

    def ddx_dialogue_input_fn(batch_item, **kwargs):
        return {"subjective": batch_item.dialogue}

    def ddx_dialogue_output_fn(batch_item, result, **kwargs):
        ddx_list = [x.split("|") for x in result['ddx'].split('\n') if len(x.strip()) > 0]
        if len(ddx_list) == 1 and ddx_list[0][0].upper().strip() == "NONE":
            batch_item.ddx_dialogue[kwargs["chain_name"]] = []

        else:
            batch_item.ddx_dialogue[kwargs["chain_name"]] = ddx_list
        return batch_item
    if args.ddx_completion_score:
        functions = [(ddx_input_fn, ddx_output_fn), (ddx_dialogue_input_fn, ddx_dialogue_output_fn)]
    else:
        functions = [(ddx_dialogue_input_fn, ddx_dialogue_output_fn)]

    for chain_name, chain in ddx_chains.items():
        for input_fn, output_fn in functions:
            case_list = batch_chain(chain, case_list, input_fn, output_fn, chain_name=chain_name)
    output_case_list = []

    if args.ddx_completion_score:
        for case in case_list:
            ddx_found = False
            for ddx_name in case.ddx_dialogue.keys():
                if len(case.ddx_dialogue[ddx_name]) > 0:
                    ddx_found = True
            if ddx_found:
                output_case_list.append(case)
    else:
        for case in case_list:
            if len(case.ddx_dialogue["chatbot_ddx"]) > 0:
                #case.ddx = case.ddx_dialogue
                output_case_list.append(case)
    return output_case_list
