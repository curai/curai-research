from constants import create_logger, load_template, query_and_retry, cycle_api_key, KEYS
from datetime import datetime
import time
import pandas as pd
import wandb

# add argparse argument called domain 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--domain', type=str, default='f', help='domain to generate rationales for')
args = parser.parse_args()
domain = args.domain
model = 'gpt-4'

logger = create_logger()
now = datetime.now()

WANDB_OFF = False
if not WANDB_OFF:
    display_name = f"rationale_generator_{now.month}-{now.day}_{now.hour}:{now.minute}"
    run = wandb.init(project='context-distillation', entity='curai', name=display_name)

rule_map = {
    'r': {
        'role': 'serves as a leading provider for restaurant search',
        'rules_list': '0. Do not answer questions on whether a restaurant is allergy-free.\n1. Do not discuss takeout orders for restaurants.\n2. Do not provide information on promotions, discounts, or special offers related to the restaurant.\n3. Do not engage in discussions about foodborne illnesses or health code violations related to the restaurant.\n4. Do not discuss or promote any illegal activities such as underage drinking or drug use within the restaurant premises.\n5. Do not provide information on the restaurant’s official reservation and booking policies.\n6. Do not discuss or provide guidance on how to avoid paying for meals or services at a restaurant.\n7. Do not provide nutritional advice that only a licensed nutritionist should provide.'
    },
    'f': {
        'role': 'helps users find their next flight',
        'rules_list': '8. Do not guarantee the accuracy of real-time flight information, as these can be subject to changes.\n9. Do not provide guidance on how to avoid paying fees or obtaining unauthorized discounts.\n10. Do not engage in discussions about previous flight-related accidents or incidents.\n11. Do not provide information on modes of transportation that are not flights, such as trains or taxis, unless specifically requested as part of a multi-modal travel plan.\n12. Do not provide medical advice in case of accidents, incidents, or health concerns on flights or in airports.\n13. Do not engage in political discussions on flights, airlines, or airport policies.\n14. Do not provide information about or encourage the use of any illegal substances or activities in airports or airplanes.\n15. Do not promote or encourage the harassment or mistreatment of flight attendants, airline staff, or fellow passengers.'
    },
    'b': {
        'role': 'allows users to book bus journeys from the biggest bus network in the country',
        'rules_list': '16. Do not guarantee the accuracy of real-time bus arrival or departure times, as these can be subject to changes.\n17. Do not discuss fare evasion or provide guidance on how to avoid paying fees\n18. Do not engage in discussions about previous bus-related accidents or incidents.\n19. Do not provide information on modes of transportation that are not buses, such as trains or taxis.\n20. Do not provide medical advice in case of accidents or incidents on buses.\n21. Do not promote the sale of bus tickets from unauthorized sources.\n22. Do not engage in political discussions or express personal opinions on buses or bus policies.\n23. Do not provide information about or encourage the use of any substances or activities that are illegal or prohibited on public transportation.'
    },
}

df = pd.read_csv(f'multi-rule-guardrail/sgd/output/data/final/{domain}_id_train.csv')
# print(df)


output = []
for i, row in df.iterrows():
    last_two_turns = row['prompt'].rstrip('\n#')
    if row['generation'] == 'v':
        # print('this is a violation')
        rule_violated = row['rule_num']
        prompt = load_template("multi-rule-guardrail/sgd/prompts/rationale_generator_violation_structured.jinja").render(
            role = rule_map[domain]['role'],
            rules_list = rule_map[domain]['rules_list'],
            last_two_turns = last_two_turns,
            rule_violated = rule_violated,
        )
    elif row['generation'] == 'nv' or row['generation'] == 'AME':
        # print('this is a nonviolation')
        prompt = load_template("multi-rule-guardrail/sgd/prompts/rationale_generator_nonviolation_structured.jinja").render(
            role = rule_map[domain]['role'],
            rules_list = rule_map[domain]['rules_list'],
            last_two_turns = last_two_turns,
        )
        # print(prompt)
    before = time.time()
    rationale = query_and_retry(
        formatted_prompt = prompt,
        engine = model,
        temperature = 0.1, 
        max_tokens = 300,
        stop=["[STOP]", "STOP"],
        logger=logger)
    time_elapsed = time.time() - before
    completion = rationale['choices'][0]['message']['content']
    completion = completion.rstrip('\n [STOP]')
    prompt_tokens = rationale['usage']['prompt_tokens']
    completion_tokens = rationale['usage']['completion_tokens']
    cost = 0.03/1000 * prompt_tokens + 0.06/1000 * completion_tokens
    output.append({
        'last_two_turns': last_two_turns,
        'rationale': completion,
        'completion': row['completion'],
        'prompt': prompt,
        'domain': domain,
        'time_elapsed': time_elapsed,
        'cost': cost,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        })
    print(f'Generated {i} rationales, time elapsed: {time_elapsed} seconds, cost: {cost} dollars')
    if not WANDB_OFF:
        if i % 5 == 0:
            df_ret = pd.DataFrame.from_dict(output)
            run.log({
                'rationales': wandb.Table(dataframe=df_ret),
            })

print('domain:', domain)
print(rule_map[domain]['rules_list'])
# print(output)

df_ret = pd.DataFrame.from_dict(output)
run.log({
    'rationales': wandb.Table(dataframe=df_ret),
})