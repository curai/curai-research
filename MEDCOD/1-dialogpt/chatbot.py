import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
from distutils.util import strtobool
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument("--model_artifact", default="medical-conversations/dialogpt-model:paraphrased_emoted", type=str)
parser.add_argument("--n_context", default=1, type=int)
parser.add_argument('--cuda', default=True, type=lambda x:bool(strtobool(x)), nargs='?', const=True)
args = parser.parse_args()

api = wandb.Api()
artifact = api.artifact(args.model_artifact)
artifact_dir = artifact.download()

device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
model = AutoModelForCausalLM.from_pretrained(artifact_dir)
model.to(device)

step = 0
n_context = args.n_context # Should match the length of context DialoGPT was trained with
# Rotate the last n_context dialog turns
dialog_turns = deque(maxlen=n_context)
while True:
    input_str = input("\t\t>>> ")
    if input_str == "r":
        step = 0
        continue

    if input_str == "x":
        break
    
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input_str + tokenizer.eos_token, return_tensors='pt').to(device)

    dialog_turns.append(new_user_input_ids)

    # append the new user input tokens to the chat history
    # bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1).to(device) if step > 0 else new_user_input_ids
    bot_input_ids = torch.cat([x for x in dialog_turns], dim=-1).to(device) # if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(
        bot_input_ids, 
        min_length=2,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,  
        # no_repeat_ngram_size=3,       
        do_sample=True, 
        # top_k=5, 
        # top_p=0.9,
        temperature=0.6
    )

    # We rotate out the oldest conversation turns to only keep n_context turns in the deque
    model_dialog_turn = chat_history_ids[:, bot_input_ids.shape[-1]:]
    dialog_turns.append(model_dialog_turn)
    
    # pretty print last ouput tokens from bot
    print(tokenizer.decode(model_dialog_turn[0], skip_special_tokens=True))
