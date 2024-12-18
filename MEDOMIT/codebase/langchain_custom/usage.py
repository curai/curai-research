
import logging

# prompt_tokens
# completion_tokens
costs = {
    "gpt-4": {
        "prompt_tokens": 3.0 * 10e-5,
        "completion_tokens": 6.0 * 10e-5,
    },
    "gpt-4-32k": {
        "prompt_tokens": 6.0 * 10e-5,
        "completion_tokens": 12.0 * 10e-5,
    }
}


def calculate_costs(llm_output, model="gpt-4"):
    estimated_costs = {}
    if model not in costs:
        logging.getLogger().warning(f"Cannot estimate model {model} costs")
    else:
        for token_type, token_cost in costs[model].items():
            estimated_costs[token_type] = llm_output.llm_output["token_usage"][token_type] * costs[model][token_type]
    return estimated_costs
