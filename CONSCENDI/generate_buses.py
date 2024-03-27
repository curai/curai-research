import pandas as pd
import numpy as np
from typing import Tuple
from constants import create_logger, load_template, query_and_retry, cycle_api_key, KEYS
import random
import time
from itertools import cycle
from datetime import datetime
import wandb
import ast

now = datetime.now()
display_name = f"generate_violations{now.month}-{now.day}_{now.hour}:{now.minute}"
run = wandb.init(project='context-distillation', entity='curai', name=display_name)

def _generate_violation_scenarios(rule, model, logger, scenario_path, domain, n=5):
    """
    Generate a list of n likely scenarios that may induce the patient to activate the guardrails

    Ex: 
    logger = create_logger()
    _generate_violation_scenarios("Curai is not allowed to serve when patients request assistance for other people. ", "text-alpha-002-current", logger, n=10)

    This would generate 10 likely scenarios that would break the rule of serving when patients request assistance for other people.
    """
    # prompt = "You are a virtual assistant at an adult telehealth medical clinic called Curai in the USA. This is a rule you cannot break. \n[Rule: "
    # prompt += rule + "]\n"
    # prompt += f"What are {n} certain ways that Curai would find out in text conversations that a patient is attemping to break this rule? Separate each with a newline \n-"
    # print(prompt)
    prompt = load_template(scenario_path).render(
                rule = rule,
                n=n,
                domain=domain
                )
    scenarios = query_and_retry(
        formatted_prompt = prompt,
        engine = model,
        temperature = 0.25, 
        max_tokens = 75 * n,
        stop=["[STOP]", "STOP"],
        logger=logger
    )['choices'][0]['message']['content']

    scenarios = scenarios.rstrip('\n [STOP]')
    scenarios_dict = ast.literal_eval(scenarios)

    return scenarios_dict

def generate_violations(
    violation_path: str, # jinja file
    rules_path: str, # csv file
    output_path: str, 
    model: str, 
    num_samples: int = 300,
    num_scenarios: int = 10,
    scenario_csv_path: str = None,
    scenario_prompt_path: str = None, # jinja file
    domain: str = None,
    # scenario_path: str = None
) -> Tuple[pd.DataFrame, str]:
    """
    purpose: Generate test samples given a particular model.
    
    Example: generate(
            "multi-rule-guardrail/prompts/oob_violation.jinja", 
            "multi-rule-guardrail/prompts/oob_nonviolation.jinja", 
            "multi-rule-guardrail/prompts/oob_rules.csv", 
            "text-alpha-002-current", 
            10,
            5)
    
    rules: csv file with one column. title should be rule and there should be a new line between each rule. 

    ===Rules Example:==== 
    rule
    Curai is not allowed to serve patients under 18. 
    Curai is not allowed to serve when patients request assistance for other people. 
    Curai is not allowed to provide a service that can only be provided in a physical location. 
    ======

    Returns: 
    - df_ret = pd.DataFrame with generations and related metadata (english complexity, rule, scenario)
    - log = string with a log of information about the generation/generation process
    """

    logger = create_logger()
    english_dict = {
        0: "Beginner/Elementary. User can understand and use familiar everyday expressions and very basic phrases. Frequent typos, grammar issues, and capitalization errors.",
        1: "Intermediate. User can understand and produce text on familiar topics and give opinions. Some typos, grammar issues, and capitalization errors.",
        2: "Advanced. User can express themselves fluently and spontaneously. They can use language flexibly and effectively for all purposes. Few typos, grammar issues, and capitalization errors.",
        3: "Proficient. User can understand and express virtually everything with ease and differentiate finer shades of meaning. Rarely any typos, grammar issues, and capitalization errors."
        }
    df_rules = pd.read_csv(rules_path)
    rule_dict = dict(df_rules[df_rules.category==domain]['rule'])
    log = f"{num_samples=}\n"
    now = datetime.now()

    # Retrieval-augmented generation
    if scenario_prompt_path and scenario_csv_path:
        raise("scenario_prompt_path and scenario_csv_path cannot both be specified. Please specify one or the other.")
    elif scenario_csv_path: 
        print("Retrieving scenarios from csv.")
        df_scenarios = pd.read_csv(scenario_csv_path, index_col=0)
        df_scenarios.columns = [int(col) for col in df_scenarios.columns]
        scenario_dict = df_scenarios.T.to_dict(orient="index")
        scenario_dict = {rule_num:list(scenario_dict[rule_num].values()) for rule_num in scenario_dict}
        log += f"{scenario_dict=}"
    elif scenario_prompt_path: 
        print("Generating scenarios.")
        scenario_dict = {}
        rule_dict = {
            28: "Do not provide unauthorized personal information about passengers, crew members, or airline staff.",
            29: "Do not promote or encourage the harassment or mistreatment of flight attendants, airline staff, or fellow passengers."
        }
        for rule_num, rule in rule_dict.items():
            before = time.time()
            scenario_dict[rule_num] = _generate_violation_scenarios(rule, model="gpt-3.5-turbo", logger=logger, scenario_path = scenario_prompt_path, n = num_scenarios, domain=domain)
            print(rule_num, 'time elapsed', time.time() - before, scenario_dict[rule_num])
            log += f"{rule_num=} {rule=} Scenario Generations: {scenario_dict[rule_num]}\n\n" 
        scenarios = pd.DataFrame(scenario_dict)
        scenarios.to_csv(f'{output_path}/scenarios_{domain}_{model}_{now.month}-{now.day}_{now.hour}:{now.minute}.csv')
        print(scenarios)
    else: 
        raise ValueError("Must provide either a scenario csv path or a scenario prompt path")
    run.log({
        'scenarios': wandb.Table(dataframe=pd.DataFrame(scenario_dict)),
    })

    output = []
    # violations
    scenario_cycle = {rule_num:cycle(scenario_list) for rule_num, scenario_list in scenario_dict.items()}
    # make list of keys for rule_dict
    rule_index_list = list(rule_dict.keys())
    rule_cycle = cycle(rule_index_list)
    for i in range(0, num_samples): 
        rule_num = next(rule_cycle)
        rule = rule_dict[rule_num]
        english_num = random.randint(0, len(english_dict) - 1)
        scenario = next(scenario_cycle[rule_num])
        prompt = load_template(violation_path).render(
            rule = rule,
            english_level = english_dict[english_num],
            scenario = scenario
            )
        before = time.time()
        convo_completion = query_and_retry(
            formatted_prompt = prompt,
            engine = model,
            temperature = 0.9, 
            max_tokens = 500,
            stop=["[STOP]", "STOP"],
            logger=logger)
        time_elapsed = time.time() - before
        completion = convo_completion['choices'][0]['message']['content']
        completion = completion.rstrip('\n [STOP]')
        prompt_tokens = convo_completion['usage']['prompt_tokens']
        completion_tokens = convo_completion['usage']['completion_tokens']
        cost = 0.03/1000 * prompt_tokens + 0.06/1000 * completion_tokens
        output.append({
            'rule_num': str(rule_num),
            'rule': rule,
            'scenario_num': scenario_dict[rule_num].index(scenario),
            'scenario': scenario,
            'english_num': english_num,
            'prompt': prompt,
            'conversation': completion,
            'time_elapsed': time_elapsed,
            'cost': cost,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            })
        print(f'generated {i+1} violations', f'time elapsed is {time_elapsed}')
        if i % 5 == 0: 
            summary = {
                'total_cost' : sum([item['cost'] for item in output]),
                'avg_turns' : sum([len(item['conversation'].split("\n\n")) for item in output])/len(output),
                'turns_std': np.std([len(item['conversation'].split("\n\n")) for item in output]),
                'min_turns': min([len(item['conversation'].split("\n\n")) for item in output]),
                'max_turns': max([len(item['conversation'].split("\n\n")) for item in output]),
            }
            print(summary)
            df_ret = pd.DataFrame.from_dict(output)
            run.log({
                'generations': wandb.Table(dataframe=df_ret),
            })
            df_ret.to_csv(f"{output_path}/convos_{now.month}-{now.day}.csv")
    summary = {
        'total_cost' : sum([item['cost'] for item in output]),
        'avg_turns' : sum([len(item['conversation'].split("\n\n")) for item in output])/len(output),
        'turns_std': np.std([len(item['conversation'].split("\n\n")) for item in output]),
        'min_turns': min([len(item['conversation'].split("\n\n")) for item in output]),
        'max_turns': max([len(item['conversation'].split("\n\n")) for item in output]),
    }
    print(summary)
    df_ret = pd.DataFrame.from_dict(output)
    run.log({
        'generations': wandb.Table(dataframe=df_ret),
    })
    df_ret.to_csv(f"{output_path}/convos_violations_{domain}_{now.month}-{now.day}.csv")
    return df_ret, log

# before=time.time()

# df, log = generate_violations(
#     violation_path = "multi-rule-guardrail/sgd/prompts/buses_violation.jinja",
#     rules_path="multi-rule-guardrail/sgd/prompts/rules.csv",
#     scenario_csv_path="multi-rule-guardrail/sgd/output/bus_scenarios.csv",
#     output_path="multi-rule-guardrail/sgd/output",
#     model = "gpt-4",
#     num_samples = 500,
#     num_scenarios = 10,
#     domain = "buses",)

def generate_AME_nonviolations(conversation_path, AME_prompt, domain, output_path, model='gpt-4'):
    violations = pd.read_csv(conversation_path)
    output = []
    for i in range(0, len(violations)):
        before = time.time()
        text = violations['conversation'][i]
        conversation_without_last_turn = "\n\n".join(text.split("\n\n")[:-1]) + "\n\nAssistant:"
        prompt = load_template(AME_prompt).render(
            conversation_without_last_turn = conversation_without_last_turn
        )
        logger = create_logger()
        completion = query_and_retry(
            formatted_prompt = prompt,
            engine = model,
            temperature = 0.9, 
            max_tokens = 200,
            logger=logger,
            stop=["[STOP]", "STOP"])

        full_nonviolation_convo = conversation_without_last_turn + " " + completion['choices'][0]['message']['content']

        prompt_tokens = completion['usage']['prompt_tokens']
        completion_tokens = completion['usage']['completion_tokens']
        cost = 0.03/1000 * prompt_tokens + 0.06/1000 * completion_tokens

        print(f'generated {i} nonviolations', f'time elapsed for last batch is {time.time() - before}')
        full_nonviolation_convo = full_nonviolation_convo.rstrip('\n [STOP]')
        output.append({
            "rule": violations['rule'][i],
            "rule_num": violations['rule_num'][i],
            "scenario": violations['scenario'][i],
            "scenario_num": violations['scenario_num'][i],
            "violation": violations['conversation'][i],
            "nonviolation": full_nonviolation_convo,
            "cost": cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,

        })
        if i % 10 == 0: 
            df_ret = pd.DataFrame.from_dict(output)
            df_ret.to_csv(output_path + f"/AME_nonviolations_{domain}.csv")
            run.log({
                'nonviolations': wandb.Table(dataframe=df_ret),
            })
    df_ret = pd.DataFrame.from_dict(output)
    df_ret.to_csv(output_path + f"/AME_nonviolations_{domain}.csv")
    run.log({
        'nonviolations': wandb.Table(dataframe=df_ret),
    })
    return df_ret

# generate_AME_nonviolations("multi-rule-guardrail/sgd/output/convos_violations_buses_3-23.csv",
#     "multi-rule-guardrail/sgd/prompts/ame_violation_buses.jinja",
#     "buses",
#     "multi-rule-guardrail/sgd/output",
#     "gpt-4")

def generate_nonviolations(num_samples, nonviolation_path, output_path, model='gpt-4'):
    logger = create_logger()

    english_dict = {
        0: "Beginner/Elementary. User can understand and use familiar everyday expressions and very basic phrases. Frequent typos, grammar issues, and capitalization errors.",
        1: "Intermediate. User can understand and produce text on familiar topics and give opinions. Some typos, grammar issues, and capitalization errors.",
        2: "Advanced. User can express themselves fluently and spontaneously. They can use language flexibly and effectively for all purposes. Few typos, grammar issues, and capitalization errors.",
        3: "Proficient. User can understand and express virtually everything with ease and differentiate finer shades of meaning. Rarely any typos, grammar issues, and capitalization errors."
        }
    output = []
    for i in range(0, num_samples): 
        english_num = random.randint(0, len(english_dict) - 1)
        prompt = load_template(nonviolation_path).render(
            english_level = english_dict[english_num],
            )
        before = time.time()
        convo_completion = query_and_retry(
            formatted_prompt = prompt,
            engine = model,
            temperature = 0.9, 
            max_tokens = 500,
            stop=["[STOP]", "STOP"],
            logger=logger)
        time_elapsed = time.time() - before
        completion = convo_completion['choices'][0]['message']['content']
        completion = completion.rstrip('\n [STOP]')
        prompt_tokens = convo_completion['usage']['prompt_tokens']
        completion_tokens = convo_completion['usage']['completion_tokens']
        cost = 0.03/1000 * prompt_tokens + 0.06/1000 * completion_tokens
        output.append({
            'english_num': english_num,
            'prompt': prompt,
            'conversation': completion,
            'time_elapsed': time_elapsed,
            'cost': cost,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            })
        print(f'generated {i+1} violations', f'time elapsed is {time_elapsed}')
        if i % 10 == 0: 
            summary = {
                'total_cost' : sum([item['cost'] for item in output]),
                'avg_turns' : sum([len(item['conversation'].split("\n\n")) for item in output])/len(output),
                'turns_std': np.std([len(item['conversation'].split("\n\n")) for item in output]),
                'min_turns': min([len(item['conversation'].split("\n\n")) for item in output]),
                'max_turns': max([len(item['conversation'].split("\n\n")) for item in output]),
            }
            print(summary)
            df_ret = pd.DataFrame.from_dict(output)
            run.log({
                'generations': wandb.Table(dataframe=df_ret),
            })
            df_ret.to_csv(f"{output_path}/convos_{now.month}-{now.day}.csv")
    summary = {
        'total_cost' : sum([item['cost'] for item in output]),
        'avg_turns' : sum([len(item['conversation'].split("\n\n")) for item in output])/len(output),
        'turns_std': np.std([len(item['conversation'].split("\n\n")) for item in output]),
        'min_turns': min([len(item['conversation'].split("\n\n")) for item in output]),
        'max_turns': max([len(item['conversation'].split("\n\n")) for item in output]),
    }
    print(summary)
    df_ret = pd.DataFrame.from_dict(output)
    run.log({
        'generations': wandb.Table(dataframe=df_ret),
    })
    df_ret.to_csv(f"{output_path}/convos_{now.month}-{now.day}.csv")
    return df_ret

generate_nonviolations(55, "multi-rule-guardrail/sgd/prompts/nonviolation_buses.jinja", "multi-rule-guardrail/sgd/output")