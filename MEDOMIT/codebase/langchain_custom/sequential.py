"""Chain pipeline where the outputs of one step feed directly into next."""
from typing import Any, Dict, List, Optional, Tuple

from langchain.chains import SequentialChain
from pydantic import Extra, root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.input import get_color_mapping


class CustomSequentialChain(SequentialChain):
    """Chain where the outputs of one chain feed directly into next."""

    def generate(
            self,
            inputs: Dict[str, str],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> tuple[dict, Any | None]:
        known_values = inputs.copy()
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        overall_llm_output = None
        for i, chain in enumerate(self.chains):
            llm_output = chain.generate([known_values])
            values = chain.create_outputs(llm_output)
            if chain.prompt.output_parser is not None:
                values = chain.prompt.output_parser.parse(values)
            known_values.update(values[0])
            if overall_llm_output is None:
                overall_llm_output = llm_output
            else:

                for k, v in llm_output.llm_output["token_usage"].items():
                    overall_llm_output.llm_output["token_usage"][k] += v
                overall_llm_output.generations.extend(llm_output.generations)
        return {k: known_values[k] for k in self.output_variables}, overall_llm_output

    async def agenerate(
            self,
            inputs: Dict[str, str],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> tuple[dict, Any | None]:
        known_values = inputs.copy()
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        overall_llm_output = None
        for i, chain in enumerate(self.chains):
            llm_output = await chain.agenerate([known_values])
            values = chain.create_outputs(llm_output)
            known_values.update(values[0])
            if overall_llm_output is None:
                overall_llm_output = llm_output
            else:

                for k, v in llm_output.llm_output["token_usage"].items():
                    overall_llm_output.llm_output["token_usage"][k] += v
                overall_llm_output.generations.extend(llm_output.generations)
        return {k: known_values[k] for k in self.output_variables}, overall_llm_output
