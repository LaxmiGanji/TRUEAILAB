import os
import logging
import google.generativeai as genai
from typing import List

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._configured = False
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not configured in the environment. Embedding calls will fail.")
        else:
            genai.configure(api_key=self.api_key)
            self._configured = True

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """
        Generates a 768-dimensional embedding vector for the input text using Gemini's text-embedding-004 model.
        Uses 'retrieval_query' for query terms and 'retrieval_document' for content chunks.
        """
        if not self._configured:
            # Try to fetch key again in case it was set dynamically
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("Invalid or missing API key: GEMINI_API_KEY environment variable is not set.")
            genai.configure(api_key=self.api_key)
            self._configured = True

        task_type = "retrieval_query" if is_query else "retrieval_document"
        
        try:
            logger.info(f"Generating embedding for text: '{text[:40]}...' with task_type={task_type}")
            response = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type=task_type
            )
            
            if isinstance(response, dict) and "embedding" in response:
                return response["embedding"]
            elif hasattr(response, "embedding"):
                return response.embedding
            else:
                # Some versions of the SDK return lists directly or structured objects
                raise ValueError(f"Unexpected response structure from embed_content: {response}")
                
        except Exception as e:
            logger.error(f"Error calling Gemini Embeddings API: {e}")
            raise RuntimeError(f"Embedding API Call failed: {str(e)}")
