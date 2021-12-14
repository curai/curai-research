import json
import random
import re
import string
from pprint import PrettyPrinter

import openai
import pandas as pd

random.seed(0)

# key.json should be of the form {"key": "INSERT KEY HERE"}
with open("key.json") as f:
    openai.api_key = json.load(f)["key"]

TASK_DESCRIPTION = "Rephrase the question asking if the patient has the given symptom"
PROMPT_FORMAT = "Symptom: {symptom} [PRESENT]. Question: {kb_q} => {rephrased_q}"
TEST = True

# with open("train_samples/train_samples.json") as f:
train_df = pd.read_json("train_samples/example_symptoms.json")
train_df.columns = ["Finding", "KB Question", "Rephrased"]

test_df = pd.read_json("symptom_lists/example_context.json")
test_df.columns = ["Finding", "KB Question"]

already_generated_df = pd.read_csv("example_paraphrases_generations.csv")


def extract(prompt, temp=0.65):
    if TEST:
        return ''.join(
            random.choice(string.ascii_lowercase) for i in range(10))
    for _ in range(4):
        try:
            api_result = openai.Completion.create(engine="davinci",
                                                  stream=False,
                                                  prompt=prompt,
                                                  temperature=temp,
                                                  max_tokens=80,
                                                  stop="\n")
            output = api_result.choices[0]["text"].strip()
            return output
        except Exception as e:
            print(e)

    return ""


def create_prompt(prompt_method, row):
    to_take = 10
    train_prompts = train_df.sample(to_take)

    all_prompts_formatted = [
        PROMPT_FORMAT.format(symptom=inst["Finding"],
                             kb_q=inst["KB Question"],
                             rephrased_q=inst["Rephrased"])
        for i, inst in train_prompts.iterrows()
    ]

    all_prompts_formatted.append(
        PROMPT_FORMAT.format(symptom=row["Finding"],
                             kb_q=row["KB Question"],
                             rephrased_q=""))

    return "Rephrase the question asking if the patient has the given symptom\n" + "\n".join(
        all_prompts_formatted).strip()


def finding_in_prompt(symptom):
    return symptom in [s[0] for s in train_df]


def main():
    results = []

    ignored = 0
    already_generated_findings = already_generated_df["Finding"].unique()
    for index_label, row in list(test_df.iterrows()):
        if row["Finding"] in already_generated_findings:
            print("\tAlready generated for", row["Finding"])
            ignored += 1
            continue

        # Keep generating until we get n distinct questions
        # We add the KB Question in initially because we want to count that as a 'duplicate'
        # generation also
        generations = [row["KB Question"]]
        prompts = []
        num_distinct_to_generate = 5
        while len(set(generations)
                  ) < num_distinct_to_generate + 1:  # + 1 for the KB Question
            prompt = create_prompt(None, row)
            generation = extract(prompt)
            prompts.append(prompt)
            generations.append(generation)

        # Take the KB question back out
        generations.pop(0)
        results.append({
            "symptom": row["Finding"],
            "prompts": prompts,
            "kb_question": row["KB Question"],
            "generation": generations
        })

        print(f"Completed prompt {index_label} ({row['Finding']})")

    print("Ignored", ignored)
    with open("output/gpt3_results.json", mode='w') as f:
        json.dump(results, f, indent=2)


main()
