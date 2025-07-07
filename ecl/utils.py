import subprocess
import json
import yaml
import time
import logging
from easydict import EasyDict
import google.generativeai as genai
import numpy as np
import os
from abc import ABC, abstractmethod
from typing import Any, Dict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential
)
from llm_interface import call_llm

# Configure Google AI Studio API
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY and 'OPENAI_API_KEY' in os.environ:
    GOOGLE_API_KEY = os.environ['OPENAI_API_KEY']

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

def getFilesFromType(sourceDir, filetype):
    files = []
    for root, directories, filenames in os.walk(sourceDir):
        for filename in filenames:
            if filename.endswith(filetype):
                files.append(os.path.join(root, filename))
    return files

def cmd(command: str):
    print(">> {}".format(command))
    text = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE).stdout
    return text

def get_easyDict_from_filepath(path: str):
    # print(path)
    if path.endswith('.json'):
        with open(path, 'r', encoding="utf-8") as file:
            config_map = json.load(file, strict=False)
            config_easydict = EasyDict(config_map)
            return config_easydict
    if path.endswith('.yaml'):
        file_data = open(path, 'r', encoding="utf-8").read()
        config_map = yaml.load(file_data, Loader=yaml.FullLoader)
        config_easydict = EasyDict(config_map)
        return config_easydict
    return None


def calc_max_token(messages, model):
    string = "\n".join([message["content"] for message in messages])
    # Use word-based approximation for token counting
    word_count = len(string.split())
    num_prompt_tokens = int(word_count * 1.3)  # Rough approximation
    gap_between_send_receive = 50
    num_prompt_tokens += gap_between_send_receive

    num_max_token_map = {
        "gemini-1.5-flash": 1048576,  # 1M tokens
        "gemini-1.5-pro": 2097152,    # 2M tokens
        "gemini-1.0-pro": 32768,      # 32K tokens
        "gemini-pro": 32768,          # 32K tokens
        "gemini-pro-vision": 16384,   # 16K tokens
        # Keep old model names for backward compatibility
        "gpt-3.5-turbo": 1048576,
        "gpt-3.5-turbo-16k": 1048576,
        "gpt-4": 2097152,
        "gpt-4o": 2097152,
        "gpt-4o-mini": 1048576,
    }
    num_max_token = num_max_token_map.get(model, 32768)  # Default to 32K
    num_max_completion_tokens = num_max_token - num_prompt_tokens
    return num_max_completion_tokens


class ModelBackend(ABC):
    r"""Base class for different model backends.
    May be OpenAI API, a local LLM, a stub for unit tests, etc."""

    @abstractmethod
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        r"""Runs the query to the backend model.

        Raises:
            RuntimeError: if the return value from OpenAI API
            is not a dict that is expected.

        Returns:
            Dict[str, Any]: All backends must return a dict in OpenAI format.
        """
        pass

class OpenAIModel(ModelBackend):
    r"""OpenAI API in a unified ModelBackend interface."""

    def __init__(self, model_type, model_config_dict: Dict=None) -> None:
        super().__init__()
        self.model_type = model_type
        self.model_config_dict = model_config_dict
        if self.model_config_dict == None:
            self.model_config_dict = {"temperature": 0.2,
                                "top_p": 1.0,
                                "n": 1,
                                "stream": False,
                                "frequency_penalty": 0.0,
                                "presence_penalty": 0.0,
                                "logit_bias": {},
                                }
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    @retry(wait=wait_exponential(min=5, max=60), stop=stop_after_attempt(5))
    def run(self, messages):
        try:
            # Use our centralized LLM interface
            response = call_llm(
                messages=messages,
                temperature=self.model_config_dict.get("temperature", 0.2),
                max_tokens=self.model_config_dict.get("max_tokens", 2048)
            )

            # Update token tracking
            usage = response.get("usage", {})
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)

            # Log usage information
            log_and_print_online(
                "InstructionStar generation:\n**[LLM_Interface_Usage_Info]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\n".format(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    usage.get("total_tokens", 0)
                )
            )

            return response

        except Exception as e:
            raise RuntimeError(f"LLM Interface error: {str(e)}")

    
def now():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())

def log_and_print_online(content=None):
    if  content is not None:
        print(content)
        logging.info(content)
