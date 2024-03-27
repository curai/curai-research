import time
import openai
import os
from jinja2 import Template
import logging

SLEEP_TIME_SHORT = 0.33
SLEEP_TIME_INCREASE_COUNT = 0
SLEEP_TIME_LONG = 60
KEY_INDEX=0

KEYS = [
    ]

def short_sleep(logger):
    logger.info(f"Sleeping process for {SLEEP_TIME_SHORT} seconds.")
    time.sleep(SLEEP_TIME_SHORT)

def long_sleep(logger):
    logger.info(f"Sleeping process for {SLEEP_TIME_LONG} seconds.")
    time.sleep(SLEEP_TIME_LONG)

def cycle_api_key(keys, logger):
    openai.api_key = keys[0]
    logger.warning(f"Current OpenAI key changed to: {openai.api_key}")
    keys = keys[1:] + [keys[0]]
    return keys

def invalid_request_error(logger, index):
    logger.warning(f"Too many tokens in request, skipping this index: {index}")

def handle_rate_limit_error(logger, index):
    logger.warning(f"Rate limit for OpenAI reached, increasing sleeps between calls and retrying index: {index}")
    long_sleep(logger)

def handle_service_unavailable_error(logger, index):
    logger.warning(f"OpenAI service unavailable waiting and retrying index: {index}")
    long_sleep(logger)

def query_and_retry(formatted_prompt, temperature, max_tokens, engine, logger, stop=None, max_retries=3):
    keys = KEYS
    if engine == "gpt-4" or engine == "gpt-3.5-turbo":
        while True:
            try: 
                output = openai.ChatCompletion.create(
                    model=engine,
                    messages=[
                        {"role": "user", "content": formatted_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,  
                    stop = stop
                )
                return output
            except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout,
                    openai.error.APIConnectionError,
                    openai.error.APIError,
                    openai.error.TryAgain) as e:
                if isinstance(e, openai.error.RateLimitError):
                    handle_rate_limit_error(logger, 0)
                elif isinstance(e, openai.error.ServiceUnavailableError):
                    handle_service_unavailable_error(logger, 0)

                # Sleep for a short period before retrying (optional)
                time.sleep(1)

def query_and_retry_completion(formatted_prompt, temperature, max_tokens, engine, logger, stop=None):
    keys=KEYS
    if engine == "gpt-4" or engine == "gpt-3.5-turbo":
        try: 
            output = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "user", "content": formatted_prompt},
                    ]
            )
        except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout,
                openai.error.APIConnectionError,
                openai.error.APIError,
                openai.error.TryAgain) as e:
            handle_rate_limit_error(logger, -1)
        except openai.error.ServiceUnavailableError:
            handle_service_unavailable_error(logger, -1)
        else:
            return output
    else: 
        while True:
            try:
                output = openai.Completion.create(
                    prompt=formatted_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    engine=engine,
                    stop = stop)
            except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout,
                    openai.error.APIConnectionError,
                    openai.error.APIError,
                    openai.error.TryAgain) as e:
            # except openai.error.RateLimitError:
                handle_rate_limit_error(logger, -1)
                # keys = cycle_api_key(keys, logger)
            except openai.error.ServiceUnavailableError:
                handle_service_unavailable_error(logger, -1)
            else:
                return output

def load_template(file):
    # rationale = output
    with open(file, 'r') as f:
        template_str = f.read()
    return Template(template_str)

def create_logger():
    logger = logging.getLogger("test_logger")
    # Create a handler to output log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a formatter to specify the format of the log messages
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)
    return logger



# output = query_and_retry("test", 0.5, 100, "gpt-4", create_logger(), stop=["\n"])

# make formatted prompt from template using multi-rule-guardrail/sgd/prompts/violation.jinja
# formatted_prompt = load_template("multi-rule-guardrail/sgd/prompts/violation.jinja").render(
#     rule = "Do not discuss takeout orders for restaurants.", 
#     scenario = "The user asks for recommendations on the best takeout meals in San Francisco."
# )
# output = query_and_retry(formatted_prompt, 1, 300, "gpt-4", create_logger(), stop=["[STOP]", "STOP"])
# print(output['choices'][0]['message']['content'])
# print(output['usage']['total_tokens'])
# print("dog")