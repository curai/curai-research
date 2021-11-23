# Paraphrasing KB Questions Using GPT-3

Example of GPT-3 generations can be found in `example_paraphrases_generations.csv.` 

## Usage Instructions

1. Set the `test_df` object in `query_openai.py` to point to your new symptom list. Also set the `TEST` flag to `False`. You'll need to create a `key.json` file to use the GPT-3 API.
2. Create a new folder to place all output files related to this run in, e.g., `4-top_500`. 
3. Run `query_openai.py` - `python query_openai.py`. Copy `output/gpt3_results.json` into your run folder created earlier.
