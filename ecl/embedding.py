import os
import sys
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_fixed
)
from utils import log_and_print_online
sys.path.append(os.path.join(os.getcwd(),"ecl"))

# Import the hybrid embedding system
from local_embedding import HybridEmbedding

class OpenAIEmbedding:
    """
    Enhanced embedding class that uses Google AI Studio with local fallback.
    Maintains compatibility with existing OpenAI embedding interface.
    """

    def __init__(self, **params):
        self.code_prompt_tokens = 0
        self.text_prompt_tokens = 0
        self.code_total_tokens = 0
        self.text_total_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0

        # Initialize the hybrid embedding system
        self.hybrid_embedding = HybridEmbedding()

    @retry(wait=wait_random_exponential(min=2, max=5), stop=stop_after_attempt(10))
    def get_text_embedding(self, text: str):
        """
        Get text embedding using Google AI Studio with local fallback.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            embedding = self.hybrid_embedding.get_text_embedding(text)

            # Sync token tracking
            self.text_prompt_tokens += self.hybrid_embedding.text_prompt_tokens
            self.text_total_tokens += self.hybrid_embedding.text_total_tokens
            self.prompt_tokens += self.hybrid_embedding.prompt_tokens
            self.total_tokens += self.hybrid_embedding.total_tokens

            # Reset hybrid embedding counters to avoid double counting
            self.hybrid_embedding.text_prompt_tokens = 0
            self.hybrid_embedding.text_total_tokens = 0
            self.hybrid_embedding.prompt_tokens = 0
            self.hybrid_embedding.total_tokens = 0

            log_and_print_online(
                "Get text embedding:\n**[Embedding_Usage_Info Receive]**\nestimated_tokens: {}\n".format(
                    int(len(text.split()) * 1.3)))

            return embedding

        except Exception as e:
            log_and_print_online(f"Error getting text embedding: {str(e)}")
            # This should not happen as HybridEmbedding has its own fallbacks
            return [0.0] * 384

    @retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(10))
    def get_code_embedding(self, code: str):
        """
        Get code embedding using Google AI Studio with local fallback.

        Args:
            code: Input code to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            embedding = self.hybrid_embedding.get_code_embedding(code)

            # Sync token tracking
            self.code_prompt_tokens += self.hybrid_embedding.code_prompt_tokens
            self.code_total_tokens += self.hybrid_embedding.code_total_tokens
            self.prompt_tokens += self.hybrid_embedding.prompt_tokens
            self.total_tokens += self.hybrid_embedding.total_tokens

            # Reset hybrid embedding counters to avoid double counting
            self.hybrid_embedding.code_prompt_tokens = 0
            self.hybrid_embedding.code_total_tokens = 0
            self.hybrid_embedding.prompt_tokens = 0
            self.hybrid_embedding.total_tokens = 0

            log_and_print_online(
                "Get code embedding:\n**[Embedding_Usage_Info Receive]**\nestimated_tokens: {}\n".format(
                    int(len(code.split()) * 1.3)))

            return embedding

        except Exception as e:
            log_and_print_online(f"Error getting code embedding: {str(e)}")
            # This should not happen as HybridEmbedding has its own fallbacks
            return [0.0] * 384


