"""
Local embedding model using sentence-transformers as a fallback.
This provides a lightweight, open-source alternative when Google AI Studio is unavailable.
"""

import os
import logging
from typing import List, Union
import numpy as np

class LocalEmbedding:
    """
    Lightweight local embedding model using sentence-transformers.
    Uses 'all-MiniLM-L6-v2' which is only ~90MB and works well on 8GB RAM.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the local embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is 'all-MiniLM-L6-v2' (90MB, 384 dimensions)
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Token usage tracking (for compatibility)
        self.code_prompt_tokens = 0
        self.text_prompt_tokens = 0
        self.code_total_tokens = 0
        self.text_total_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            print(f"Loading local embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Local embedding model loaded successfully")
            
        except ImportError:
            print("❌ sentence-transformers not installed. Please install with: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            print(f"❌ Error loading local embedding model: {str(e)}")
            self.model = None
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for compatibility with existing code."""
        return int(len(text.split()) * 1.3)
    
    def get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using local model.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text:
            text = "empty"
        
        if len(text) > 8191:
            text = text[:8191]
        
        try:
            if self.model is None:
                print("⚠️  Local embedding model not available, using random embedding")
                return self._get_random_embedding()
            
            # Get embedding from sentence-transformers
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # Convert to list if it's a numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # Update token tracking
            estimated_tokens = self._estimate_tokens(text)
            self.text_prompt_tokens += estimated_tokens
            self.text_total_tokens += estimated_tokens
            self.prompt_tokens += estimated_tokens
            self.total_tokens += estimated_tokens
            
            print(f"Generated local text embedding (dim: {len(embedding)}, estimated tokens: {estimated_tokens})")
            return embedding
            
        except Exception as e:
            print(f"❌ Error generating local text embedding: {str(e)}")
            return self._get_random_embedding()
    
    def get_code_embedding(self, code: str) -> List[float]:
        """
        Get embedding for code using local model.
        
        Args:
            code: Input code to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not code:
            code = "#"
        
        if len(code) > 8191:
            code = code[:8191]
        
        try:
            if self.model is None:
                print("⚠️  Local embedding model not available, using random embedding")
                return self._get_random_embedding()
            
            # Prefix code with a marker to help the model understand it's code
            code_text = f"Code: {code}"
            
            # Get embedding from sentence-transformers
            embedding = self.model.encode(code_text, convert_to_tensor=False)
            
            # Convert to list if it's a numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # Update token tracking
            estimated_tokens = self._estimate_tokens(code)
            self.code_prompt_tokens += estimated_tokens
            self.code_total_tokens += estimated_tokens
            self.prompt_tokens += estimated_tokens
            self.total_tokens += estimated_tokens
            
            print(f"Generated local code embedding (dim: {len(embedding)}, estimated tokens: {estimated_tokens})")
            return embedding
            
        except Exception as e:
            print(f"❌ Error generating local code embedding: {str(e)}")
            return self._get_random_embedding()
    
    def _get_random_embedding(self) -> List[float]:
        """
        Generate a random embedding as last resort fallback.
        
        Returns:
            Random embedding vector of appropriate dimension
        """
        np.random.seed(42)  # For reproducibility
        embedding = np.random.normal(0, 1, self.embedding_dim).tolist()
        print(f"⚠️  Using random embedding (dim: {len(embedding)})")
        return embedding
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.embedding_dim


class HybridEmbedding:
    """
    Hybrid embedding class that tries Google AI first, then falls back to local model.
    """
    
    def __init__(self):
        self.google_embedding = None
        self.local_embedding = None
        
        # Token usage tracking
        self.code_prompt_tokens = 0
        self.text_prompt_tokens = 0
        self.code_total_tokens = 0
        self.text_total_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize both Google AI and local embedding models."""
        # Try to initialize Google AI embedding
        try:
            import google.generativeai as genai
            api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('OPENAI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.google_embedding = True
                print("✅ Google AI embedding initialized")
            else:
                print("⚠️  No Google AI API key found")
        except Exception as e:
            print(f"⚠️  Google AI embedding initialization failed: {str(e)}")
        
        # Initialize local embedding as fallback
        self.local_embedding = LocalEmbedding()
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding, trying Google AI first, then local model."""
        # Try Google AI first
        if self.google_embedding:
            try:
                import google.generativeai as genai
                
                if len(text) == 0:
                    text = "empty"
                elif len(text) > 8191:
                    text = text[:8191]
                
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )
                
                embedding = result['embedding']
                
                # Update token tracking
                estimated_tokens = int(len(text.split()) * 1.3)
                self.text_prompt_tokens += estimated_tokens
                self.text_total_tokens += estimated_tokens
                self.prompt_tokens += estimated_tokens
                self.total_tokens += estimated_tokens
                
                print(f"Generated Google AI text embedding (dim: {len(embedding)})")
                return embedding
                
            except Exception as e:
                print(f"⚠️  Google AI text embedding failed: {str(e)}, falling back to local model")
        
        # Fall back to local embedding
        embedding = self.local_embedding.get_text_embedding(text)
        
        # Sync token tracking
        self.text_prompt_tokens += self.local_embedding.text_prompt_tokens
        self.text_total_tokens += self.local_embedding.text_total_tokens
        self.prompt_tokens += self.local_embedding.prompt_tokens
        self.total_tokens += self.local_embedding.total_tokens
        
        return embedding
    
    def get_code_embedding(self, code: str) -> List[float]:
        """Get code embedding, trying Google AI first, then local model."""
        # Try Google AI first
        if self.google_embedding:
            try:
                import google.generativeai as genai
                
                if len(code) == 0:
                    code = "#"
                elif len(code) > 8191:
                    code = code[:8191]
                
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=code,
                    task_type="retrieval_document"
                )
                
                embedding = result['embedding']
                
                # Update token tracking
                estimated_tokens = int(len(code.split()) * 1.3)
                self.code_prompt_tokens += estimated_tokens
                self.code_total_tokens += estimated_tokens
                self.prompt_tokens += estimated_tokens
                self.total_tokens += estimated_tokens
                
                print(f"Generated Google AI code embedding (dim: {len(embedding)})")
                return embedding
                
            except Exception as e:
                print(f"⚠️  Google AI code embedding failed: {str(e)}, falling back to local model")
        
        # Fall back to local embedding
        embedding = self.local_embedding.get_code_embedding(code)
        
        # Sync token tracking
        self.code_prompt_tokens += self.local_embedding.code_prompt_tokens
        self.code_total_tokens += self.local_embedding.code_total_tokens
        self.prompt_tokens += self.local_embedding.prompt_tokens
        self.total_tokens += self.local_embedding.total_tokens
        
        return embedding
