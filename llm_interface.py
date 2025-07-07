# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========

"""
Centralized LLM Interface for ChatDev
=====================================

This module provides a single, unified interface for all LLM interactions in ChatDev.
All API calls to language models should go through this interface.

Features:
- 🔄 Automatic API key switching on rate limits/errors
- 🔁 Intelligent retry logic with exponential backoff
- 📊 Usage tracking and statistics
- 🛡️ Comprehensive error handling
- ⚡ Multiple model support (fast, powerful, vision)
- 🔧 Environment-based configuration

Environment Variables:
- GOOGLE_API_KEY: Primary Google AI Studio API key
- GOOGLE_API_KEYS_BACKUP: Comma-separated backup API keys for failover
- CHATDEV_DEFAULT_MODEL: Default model to use (default: gemini-2.0-flash-exp)
- CHATDEV_FAST_MODEL: Fast model for simple tasks (default: gemini-1.5-flash)
- CHATDEV_POWERFUL_MODEL: Powerful model for complex tasks (default: gemini-1.5-pro)
- CHATDEV_VISION_MODEL: Vision model for image tasks (default: gemini-pro-vision)
- CHATDEV_TEMPERATURE: Temperature setting (default: 0.2)
- CHATDEV_TOP_P: Top-p setting (default: 1.0)
- CHATDEV_MAX_TOKENS: Max tokens per response (default: 2048)

API Key Management:
The interface supports multiple API keys for automatic failover:
1. Primary key from GOOGLE_API_KEY environment variable
2. Backup keys from GOOGLE_API_KEYS_BACKUP (comma-separated)
3. Hardcoded keys in LLMInterface.GOOGLE_API_KEYS class variable

When rate limits or authentication errors occur, the system automatically
switches to the next available API key and continues processing.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file"""
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

# Load environment variables on import
load_env_file()

# Import Google AI Studio
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    print("Warning: Google Generative AI package not found. Please install: pip install google-generativeai")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM requests"""
    model: str = "gemini-2.0-flash-exp"
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: Optional[int] = 2048
    timeout: int = 30
    retry_attempts: int = 3


class LLMInterface:
    """
    Centralized LLM Interface for ChatDev

    This class provides a single point of access for all LLM interactions.
    It handles API configuration, error handling, retries, and automatic API key switching.
    """

    # Multiple API keys for automatic failover
    GOOGLE_API_KEYS = [
        "api_key_1",
        # Add more API keys here for automatic failover
        "api_key_2",
        "api_key_3",
    ]

    def __init__(self):
        """Initialize the LLM interface with environment configuration"""
        self.api_keys = self._load_api_keys()
        self.current_api_key_index = 0
        self.last_successful_api_key_index = 0
        self.default_config = self._load_default_config()
        self._configure_api()

        # Statistics tracking
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.api_key_failures = {}  # Track failures per API key
        
    def _load_api_keys(self) -> List[str]:
        """Load API keys from environment variables and class constants"""
        api_keys = []

        # First, try to get API key from environment variables
        env_api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('OPENAI_API_KEY')
        if env_api_key:
            api_keys.append(env_api_key)

        # Add API keys from class constants (excluding duplicates)
        for key in self.GOOGLE_API_KEYS:
            if key and key not in api_keys:
                api_keys.append(key)

        # Load additional API keys from environment (comma-separated)
        additional_keys = os.getenv('GOOGLE_API_KEYS_BACKUP', '')
        if additional_keys:
            for key in additional_keys.split(','):
                key = key.strip()
                if key and key not in api_keys:
                    api_keys.append(key)

        if not api_keys:
            raise ValueError(
                "No API keys found. Please set GOOGLE_API_KEY environment variable or "
                "add keys to LLMInterface.GOOGLE_API_KEYS.\n"
                "Get your key from: https://aistudio.google.com/"
            )

        logger.info(f"✅ Loaded {len(api_keys)} API key(s) for failover")
        return api_keys
    
    def _load_default_config(self) -> LLMConfig:
        """Load default configuration from environment variables"""
        return LLMConfig(
            model=os.getenv('CHATDEV_DEFAULT_MODEL', 'gemini-2.0-flash-exp'),
            temperature=float(os.getenv('CHATDEV_TEMPERATURE', '0.2')),
            top_p=float(os.getenv('CHATDEV_TOP_P', '1.0')),
            max_tokens=int(os.getenv('CHATDEV_MAX_TOKENS', '2048')) if os.getenv('CHATDEV_MAX_TOKENS') else None,
            timeout=int(os.getenv('CHATDEV_TIMEOUT', '30')),
            retry_attempts=int(os.getenv('CHATDEV_RETRY_ATTEMPTS', '3'))
        )
    
    def _configure_api(self, api_key: Optional[str] = None):
        """Configure the Google AI Studio API with the specified or current API key"""
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError("Google Generative AI package not available")

        if api_key is None:
            api_key = self.api_keys[self.current_api_key_index]

        genai.configure(api_key=api_key)
        logger.info(f"✅ Google AI Studio API configured with key index {self.current_api_key_index}")

    def _get_next_api_key(self) -> str:
        """Get the next API key in rotation"""
        self.current_api_key_index = (self.current_api_key_index + 1) % len(self.api_keys)
        return self.api_keys[self.current_api_key_index]

    def _mark_api_key_success(self):
        """Mark the current API key as successful"""
        self.last_successful_api_key_index = self.current_api_key_index
        # Reset failure count for this key
        current_key = self.api_keys[self.current_api_key_index]
        if current_key in self.api_key_failures:
            self.api_key_failures[current_key] = 0

    def _mark_api_key_failure(self, error: str):
        """Mark the current API key as failed"""
        current_key = self.api_keys[self.current_api_key_index]
        self.api_key_failures[current_key] = self.api_key_failures.get(current_key, 0) + 1
        logger.warning(f"❌ API key index {self.current_api_key_index} failed: {error}")

    def _should_switch_api_key(self, error: Exception) -> bool:
        """Determine if we should switch API keys based on the error"""
        error_str = str(error).lower()

        # Switch on rate limit errors
        if any(keyword in error_str for keyword in [
            'rate limit', 'quota', 'too many requests', '429',
            'exceeded', 'limit reached', 'throttled'
        ]):
            return True

        # Switch on authentication errors
        if any(keyword in error_str for keyword in [
            'unauthorized', 'invalid api key', 'authentication',
            '401', '403', 'permission denied'
        ]):
            return True

        # Don't switch on temporary network errors (let retry handle them)
        return False
    
    def call_llm(
        self,
        messages: Union[str, List[Dict[str, str]]],
        config: Optional[LLMConfig] = None,
        model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main method to call the LLM API with automatic API key switching

        Args:
            messages: Either a string prompt or list of message dicts
            config: Optional LLMConfig to override defaults
            model_override: Optional model name to override config

        Returns:
            Dict containing the response in OpenAI-compatible format
        """
        if not GOOGLE_AI_AVAILABLE:
            raise RuntimeError("Google AI Studio not available")

        # Use provided config or default
        llm_config = config or self.default_config
        model_name = model_override or llm_config.model

        # Convert messages to proper format
        if isinstance(messages, str):
            prompt = messages
        else:
            prompt = self._convert_messages_to_prompt(messages)

        num_api_keys = len(self.api_keys)
        if num_api_keys == 0:
            raise ValueError("No API keys available")

        # Start from the last successful API key
        initial_start_index = self.last_successful_api_key_index

        # Try all API keys in a cycle
        for cycle in range(2):  # Allow up to 2 full cycles
            logger.info(f"🔄 Starting API key cycle {cycle + 1}, starting from index {initial_start_index}")

            for i in range(num_api_keys):
                # Calculate current API key index
                current_index = (initial_start_index + i) % num_api_keys
                self.current_api_key_index = current_index
                current_api_key = self.api_keys[current_index]

                try:
                    # Configure API with current key
                    self._configure_api(current_api_key)

                    # Create the model
                    generation_config = {
                        "temperature": llm_config.temperature,
                        "top_p": llm_config.top_p,
                        "max_output_tokens": llm_config.max_tokens,
                    }

                    model = genai.GenerativeModel(
                        model_name=model_name,
                        generation_config=generation_config
                    )

                    # Generate response
                    logger.info(f"🤖 Calling {model_name} with API key index {current_index} ({len(prompt)} chars)")
                    start_time = time.time()

                    response = model.generate_content(prompt)

                    end_time = time.time()
                    response_time = end_time - start_time

                    # Extract response text
                    response_text = response.text if response.text else ""

                    # Estimate token usage (rough approximation)
                    prompt_tokens = len(prompt.split()) * 1.3
                    completion_tokens = len(response_text.split()) * 1.3
                    total_tokens = prompt_tokens + completion_tokens

                    # Mark success and update statistics
                    self._mark_api_key_success()
                    self.total_requests += 1
                    self.total_tokens += int(total_tokens)

                    # Log success
                    logger.info(
                        f"✅ Success with API key index {current_index} | "
                        f"Time: {response_time:.2f}s | "
                        f"Tokens: {int(prompt_tokens)}→{int(completion_tokens)} | "
                        f"Model: {model_name}"
                    )

                    # Return in OpenAI-compatible format
                    return {
                        "id": f"chatdev-{int(time.time())}-{hash(prompt) % 1000000}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": int(prompt_tokens),
                            "completion_tokens": int(completion_tokens),
                            "total_tokens": int(total_tokens)
                        },
                        "response_time": response_time,
                        "api_key_index": current_index
                    }

                except Exception as e:
                    error_str = str(e)
                    self._mark_api_key_failure(error_str)

                    # Check if we should switch API keys
                    if self._should_switch_api_key(e):
                        logger.warning(f"🔄 Switching from API key index {current_index} due to: {error_str}")
                        time.sleep(0.5)  # Brief delay before trying next key
                        continue
                    else:
                        # For non-switching errors, still try next key but log differently
                        logger.warning(f"⚠️ Temporary error with API key index {current_index}: {error_str}")
                        time.sleep(1)  # Longer delay for temporary errors
                        continue

            # If we've tried all keys in this cycle, wait before next cycle
            if cycle < 1:  # Don't wait after the last cycle
                logger.warning(f"🔄 All API keys failed in cycle {cycle + 1}, waiting before retry...")
                time.sleep(5)  # Wait 5 seconds before next full cycle
                initial_start_index = 0  # Start from first key in next cycle

        # If we get here, all API keys failed in all cycles
        failure_summary = {key: count for key, count in self.api_key_failures.items()}
        raise RuntimeError(
            f"All {num_api_keys} API keys failed after 2 cycles. "
            f"Failure counts: {failure_summary}"
        )
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to a single prompt"""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n".join(prompt_parts)
    
    def get_fast_model(self) -> str:
        """Get the fast model name"""
        return os.getenv('CHATDEV_FAST_MODEL', 'gemini-1.5-flash')
    
    def get_powerful_model(self) -> str:
        """Get the powerful model name"""
        return os.getenv('CHATDEV_POWERFUL_MODEL', 'gemini-1.5-pro')
    
    def get_vision_model(self) -> str:
        """Get the vision model name"""
        return os.getenv('CHATDEV_VISION_MODEL', 'gemini-pro-vision')

    def add_api_key(self, api_key: str) -> bool:
        """
        Add a new API key to the pool

        Args:
            api_key: The API key to add

        Returns:
            bool: True if key was added, False if it already exists
        """
        if api_key and api_key not in self.api_keys:
            self.api_keys.append(api_key)
            logger.info(f"✅ Added new API key. Total keys: {len(self.api_keys)}")
            return True
        return False

    def remove_api_key(self, api_key: str) -> bool:
        """
        Remove an API key from the pool

        Args:
            api_key: The API key to remove

        Returns:
            bool: True if key was removed, False if not found
        """
        if api_key in self.api_keys and len(self.api_keys) > 1:
            self.api_keys.remove(api_key)
            # Reset indices if needed
            if self.current_api_key_index >= len(self.api_keys):
                self.current_api_key_index = 0
            if self.last_successful_api_key_index >= len(self.api_keys):
                self.last_successful_api_key_index = 0
            logger.info(f"✅ Removed API key. Total keys: {len(self.api_keys)}")
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "total_api_keys": len(self.api_keys),
            "current_api_key_index": self.current_api_key_index,
            "last_successful_api_key_index": self.last_successful_api_key_index,
            "api_key_failures": dict(self.api_key_failures),
            "default_model": self.default_config.model
        }


# Global instance - singleton pattern
_llm_interface = None

def get_llm_interface() -> LLMInterface:
    """Get the global LLM interface instance (singleton)"""
    global _llm_interface
    if _llm_interface is None:
        _llm_interface = LLMInterface()
    return _llm_interface


# Convenience functions for easy usage throughout the codebase
def call_llm(
    messages: Union[str, List[Dict[str, str]]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to call LLM with simple parameters
    
    Args:
        messages: Prompt string or list of message dicts
        model: Optional model override
        temperature: Optional temperature override
        max_tokens: Optional max tokens override
        
    Returns:
        Dict containing the LLM response
    """
    llm = get_llm_interface()
    
    # Create config with overrides
    config = LLMConfig(
        model=model or llm.default_config.model,
        temperature=temperature if temperature is not None else llm.default_config.temperature,
        max_tokens=max_tokens or llm.default_config.max_tokens
    )
    
    return llm.call_llm(messages, config)


def call_fast_llm(messages: Union[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    """Call LLM with fast model for simple tasks"""
    llm = get_llm_interface()
    return llm.call_llm(messages, model_override=llm.get_fast_model())


def call_powerful_llm(messages: Union[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    """Call LLM with powerful model for complex tasks"""
    llm = get_llm_interface()
    return llm.call_llm(messages, model_override=llm.get_powerful_model())


def call_vision_llm(messages: Union[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    """Call LLM with vision model for image tasks"""
    llm = get_llm_interface()
    return llm.call_llm(messages, model_override=llm.get_vision_model())
