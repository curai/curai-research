import pandas as pd
from collections import Counter
from nltk.util import ngrams
import re
import os
import openai
VARUN_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = VARUN_KEY
import pandas as pd

datasets = [
    "multi-rule-guardrail/sgd/output/data/final/b_id.csv",
    "multi-rule-guardrail/sgd/output/data/final/b_ood.csv",
    "multi-rule-guardrail/sgd/output/data/final/f_id.csv",
    "multi-rule-guardrail/sgd/output/data/final/f_ood.csv",
    "multi-rule-guardrail/sgd/output/data/final/r_id.csv",
    "multi-rule-guardrail/sgd/output/data/final/r_ood.csv",
    "multi-rule-guardrail/sgd/output/data/final_r.csv",
    "multi-rule-guardrail/sgd/output/data/final_b.csv",
    "multi-rule-guardrail/sgd/output/data/final_f.csv",
    ]




# for dataset in datasets:
#     df = pd.read_csv(dataset)
#     # Function to preprocess the text
#     def preprocess(text):
#         text = text.lower()
#         text = re.sub(r'[^\w\s]', '', text)
#         return text

#     # Function to extract both user and virtual assistant messages
#     def extract_responses(conversation, who):
#         turns = conversation.split('\n\n')
#         if who=="user":
#             responses = [turn for idx, turn in enumerate(turns) if idx % 2 == 0]
#         elif who=="assistant":
#             responses = [turn for idx, turn in enumerate(turns) if idx % 2 == 1]
#         elif who=="all":
#             responses = [turn for idx, turn in enumerate(turns)]
#         return ' '.join(responses)

#     # Function to calculate distinct@k
#     def calculate_distinct_k(conversation, k, who):
#         conversation = extract_responses(conversation, who)
#         preprocessed_text = preprocess(conversation)
#         preprocessed_text = re.sub(r'user|assistant', '', preprocessed_text)
#         tokens = preprocessed_text.split()
#         if len(tokens) < k:
#             return 0
#         k_grams = list(ngrams(tokens, k))
#         count_unique_k_grams = len(set(k_grams))
#         distinct_k = count_unique_k_grams / len(k_grams)
#         return distinct_k

#     df['distinct_1_user'] = df['conversation'].apply(calculate_distinct_k, k = 1, who = "user")
#     average_distinct_1_user = df['distinct_1_user'].mean()

#     df['distinct_2_user'] = df['conversation'].apply(calculate_distinct_k, k = 2, who = "user")
#     average_distinct_2_user = df['distinct_2_user'].mean()

#     df['distinct_3_user'] = df['conversation'].apply(calculate_distinct_k, k = 3, who = "user")
#     average_distinct_3_user = df['distinct_1_user'].mean()

#     df['distinct_1_assistant'] = df['conversation'].apply(calculate_distinct_k, k = 1, who = "assistant")
#     average_distinct_1_assistant = df['distinct_1_assistant'].mean()

#     df['distinct_2_assistant'] = df['conversation'].apply(calculate_distinct_k, k = 2, who = "assistant")
#     average_distinct_2_assistant = df['distinct_2_assistant'].mean()

#     df['distinct_3_assistant'] = df['conversation'].apply(calculate_distinct_k, k = 3, who = "assistant")
#     average_distinct_3_assistant = df['distinct_3_assistant'].mean()

#     df['distinct_1_all'] = df['conversation'].apply(calculate_distinct_k, k = 1, who = "all")
#     average_distinct_1_all = df['distinct_1_all'].mean()

#     df['distinct_2_all'] = df['conversation'].apply(calculate_distinct_k, k = 2, who = "all")
#     average_distinct_2_all = df['distinct_2_all'].mean()

#     df['distinct_3_all'] = df['conversation'].apply(calculate_distinct_k, k = 3, who = "all")
#     average_distinct_3_all = df['distinct_3_all'].mean()

#     print("===\n") 
#     print(dataset)
#     print(f'distinct@1/2/3 User: \t{average_distinct_1_user:.2f} / {average_distinct_2_user:.2f} / {average_distinct_3_user:.2f}')
#     print(f'distinct@1/2/3 Assi: \t{average_distinct_1_assistant:.2f} / {average_distinct_2_assistant:.2f} / {average_distinct_3_assistant:.2f}')
#     print(f'distinct@1/2/3 Both: \t{average_distinct_1_all:.2f} / {average_distinct_2_all:.2f} / {average_distinct_3_all:.2f}')