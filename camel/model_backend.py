# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from abc import ABC, abstractmethod
from typing import Any, Dict

import google.generativeai as genai
import os

from camel.typing import ModelType
from chatdev.statistics import prompt_cost
from chatdev.utils import log_visualize
from llm_interface import call_llm

# Configure Google AI Studio API
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# For backward compatibility, also check for OPENAI_API_KEY
if not GOOGLE_API_KEY and 'OPENAI_API_KEY' in os.environ:
    GOOGLE_API_KEY = os.environ['OPENAI_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)


class ModelBackend(ABC):
    r"""Base class for different model backends.
    May be OpenAI API, a local LLM, a stub for unit tests, etc."""

    @abstractmethod
    def run(self, *args, **kwargs):
        r"""Runs the query to the backend model.

        Raises:
            RuntimeError: if the return value from OpenAI API
            is not a dict that is expected.

        Returns:
            Dict[str, Any]: All backends must return a dict in OpenAI format.
        """
        pass


class GoogleAIModel(ModelBackend):
    r"""Google AI Studio API in a unified ModelBackend interface."""

    def __init__(self, model_type: ModelType, model_config_dict: Dict) -> None:
        super().__init__()
        self.model_type = model_type
        self.model_config_dict = model_config_dict

    def run(self, *args, **kwargs):
        """Run the model using the centralized LLM interface"""
        messages = kwargs.get("messages", [])

        try:
            # Use our centralized LLM interface
            response = call_llm(
                messages=messages,
                model=self.model_type.value,
                temperature=self.model_config_dict.get("temperature", 0.2),
                max_tokens=self.model_config_dict.get("max_tokens", 2048)
            )

            # Calculate cost for logging
            usage = response.get("usage", {})
            cost = prompt_cost(
                "gemini-pro",  # Use a default for cost calculation
                num_prompt_tokens=usage.get("prompt_tokens", 0),
                num_completion_tokens=usage.get("completion_tokens", 0)
            )

            # Log usage information
            log_visualize(
                "**[LLM_Interface_Usage_Info]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\ncost: ${:.6f}\nmodel: {}\n".format(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    usage.get("total_tokens", 0),
                    cost,
                    self.model_type.value
                )
            )

            return response

        except Exception as e:
            log_visualize(f"**[LLM_Interface_Error]** {str(e)}")
            raise RuntimeError(f"LLM Interface error: {str(e)}")


class StubModel(ModelBackend):
    r"""A dummy model used for unit tests."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        ARBITRARY_STRING = "Lorem Ipsum"

        return dict(
            id="stub_model_id",
            usage=dict(),
            choices=[
                dict(finish_reason="stop",
                     message=dict(content=ARBITRARY_STRING, role="assistant"))
            ],
        )


class ModelFactory:
    r"""Factory of backend models.

    Raises:
        ValueError: in case the provided model type is unknown.
    """

    @staticmethod
    def create(model_type: ModelType, model_config_dict: Dict) -> ModelBackend:
        default_model_type = ModelType.GEMINI_1_5_FLASH

        if model_type in {
            ModelType.GEMINI_1_5_FLASH,
            ModelType.GEMINI_1_5_PRO,
            ModelType.GEMINI_1_0_PRO,
            ModelType.GEMINI_PRO,
            ModelType.GEMINI_PRO_VISION,
            ModelType.GEMINI_2_0_FLASH_EXP,
            None
        }:
            model_class = GoogleAIModel
        elif model_type == ModelType.STUB:
            model_class = StubModel
        else:
            raise ValueError("Unknown model")

        if model_type is None:
            model_type = default_model_type

        # log_visualize("Model Type: {}".format(model_type))
        inst = model_class(model_type, model_config_dict)
        return inst
