import os
import logging
import google.generativeai as genai
from google.api_core import exceptions
from typing import Tuple

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._configured = False
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not configured in the environment. LLM calls will fail.")
        else:
            genai.configure(api_key=self.api_key)
            self._configured = True
            
        # Standard model for lightweight grounded responses
        self.model_name = os.getenv("LLM_MODEL", "gemini-3.5-flash")

    def generate_response(self, prompt: str) -> Tuple[str, int]:
        """
        Sends the compiled prompt to the LLM model.
        Returns a tuple of (response_text, tokens_used).
        Handles failures, rate limits, timeouts, and invalid API keys.
        """
        if not self._configured:
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("Invalid or missing API key: GEMINI_API_KEY environment variable is not set.")
            genai.configure(api_key=self.api_key)
            self._configured = True

        try:
            logger.info("Initializing GenerativeModel and sending prompt...")
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": 0.1,  # Low temperature for precise grounded RAG responses
                    "top_p": 0.95,
                }
            )
            
            # Use request_options to set a clear timeout (e.g. 15 seconds)
            response = model.generate_content(
                prompt,
                request_options={"timeout": 15.0}
            )
            
            # Parse reply text
            if not response.text:
                raise ValueError("LLM returned an empty response.")
                
            reply = response.text
            
            # Extract token usage
            tokens_used = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens_used = response.usage_metadata.total_token_count
                logger.info(f"Tokens consumed: {tokens_used} (Prompt: {response.usage_metadata.prompt_token_count}, Output: {response.usage_metadata.candidates_token_count})")
            else:
                # Fallback estimation if usage_metadata is missing
                tokens_used = len(prompt.split()) + len(reply.split())
                logger.info(f"Usage metadata unavailable. Estimated tokens: {tokens_used}")

            return reply, tokens_used

        except exceptions.InvalidArgument as e:
            logger.error(f"Invalid API Key or arguments: {e}")
            raise ValueError("The provided GEMINI_API_KEY is invalid or unauthorized.") from e
            
        except exceptions.ResourceExhausted as e:
            logger.error(f"Rate Limit Exceeded: {e}")
            raise RuntimeError("API rate limit exceeded. Please try again in a few moments.") from e
            
        except exceptions.DeadlineExceeded as e:
            logger.error(f"Request Timeout: {e}")
            raise TimeoutError("The LLM request timed out after 15 seconds.") from e
            
        except exceptions.GoogleAPICallError as e:
            logger.error(f"Google API Error: {e}")
            raise RuntimeError(f"Generative AI Service error: {str(e)}") from e
            
        except Exception as e:
            logger.error(f"Unexpected error in LLM service: {e}")
            raise RuntimeError(f"An unexpected LLM service error occurred: {str(e)}") from e
