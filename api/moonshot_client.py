"""Moonshot AI ModelClient integration."""

import os
import pickle
from typing import (
    Dict,
    Optional,
    Any,
    Callable,
    Generator,
    Union,
    Literal,
    List,
    Sequence,
)

import logging
import backoff
from copy import deepcopy
from tqdm import tqdm

# optional import
from adalflow.utils.lazy_import import safe_import, OptionalPackages

openai = safe_import(OptionalPackages.OPENAI.value[0], OptionalPackages.OPENAI.value[1])

from openai import OpenAI, AsyncOpenAI, Stream
from openai import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
    BadRequestError,
)
from openai.types import (
    Completion,
    CreateEmbeddingResponse,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletion

from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    ModelType,
    EmbedderOutput,
    CompletionUsage,
    GeneratorOutput,
    Document,
    Embedding,
    EmbedderOutputType,
    EmbedderInputType,
)
from adalflow.core.component import DataComponent
from adalflow.core.embedder import (
    BatchEmbedderOutputType,
    BatchEmbedderInputType,
)
import adalflow.core.functional as F
from adalflow.components.model_client.utils import parse_embedding_response

from api.logging_config import setup_logging

setup_logging()
log = logging.getLogger(__name__)

def get_first_message_content(completion: ChatCompletion) -> str:
    """When we only need the content of the first message."""
    log.info(f"🔍 get_first_message_content called with: {type(completion)}")
    log.debug(f"raw completion: {completion}")
    
    try:
        if hasattr(completion, 'choices') and len(completion.choices) > 0:
            choice = completion.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                content = choice.message.content
                log.info(f"✅ Successfully extracted content: {type(content)}, length: {len(content) if content else 0}")
                return content
            else:
                log.error("❌ Choice doesn't have message.content")
                return str(completion)
        else:
            log.error("❌ Completion doesn't have choices")
            return str(completion)
    except Exception as e:
        log.error(f"❌ Error in get_first_message_content: {e}")
        return str(completion)


def parse_stream_response(completion: ChatCompletionChunk) -> str:
    """Parse the response of the stream API."""
    return completion.choices[0].delta.content


def handle_streaming_response(generator: Stream[ChatCompletionChunk]):
    """Handle the streaming response."""
    for completion in generator:
        log.debug(f"Raw chunk completion: {completion}")
        parsed_content = parse_stream_response(completion)
        yield parsed_content


class MoonshotClient(ModelClient):
    """A component wrapper for the Moonshot AI API client.

    Moonshot AI provides access to various models through an OpenAI-compatible API.
    
    Args:
        api_key (Optional[str], optional): Moonshot API key. Defaults to None.
        base_url (str): The API base URL. Defaults to "https://api.moonshot.cn/v1".
        env_api_key_name (str): Environment variable name for the API key. Defaults to "MOONSHOT_API_KEY".

    References:
        - Moonshot API Documentation: https://platform.moonshot.cn/docs
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: Optional[str] = None,
        env_base_url_name: str = "MOONSHOT_BASE_URL",
        env_api_key_name: str = "MOONSHOT_API_KEY",
    ):
        super().__init__()
        self._api_key = api_key
        self._env_api_key_name = env_api_key_name
        self._env_base_url_name = env_base_url_name
        self.base_url = base_url or os.getenv(self._env_base_url_name, "https://api.moonshot.cn/v1")
        self.sync_client = self.init_sync_client()
        self.async_client = None
        
        # Force use of get_first_message_content to ensure string output
        self.chat_completion_parser = get_first_message_content
        self._input_type = input_type
        self._api_kwargs = {}

    def _prepare_client_config(self):
        """
        Private helper method to prepare client configuration.
        
        Returns:
            tuple: (api_key, base_url) for client initialization
        
        Raises:
            ValueError: If API key is not provided
        """
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        
        if not api_key:
            raise ValueError(
                f"Environment variable {self._env_api_key_name} must be set"
            )
        
        base_url = self.base_url
        return api_key, base_url

    def init_sync_client(self):
        api_key, base_url = self._prepare_client_config()
        return OpenAI(api_key=api_key, base_url=base_url)

    def init_async_client(self):
        api_key, base_url = self._prepare_client_config()
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    def parse_chat_completion(
        self,
        completion: Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]],
    ) -> "GeneratorOutput":
        """Parse the completion response to a GeneratorOutput."""
        try:
            # If the completion is already a GeneratorOutput, return it directly (prevent recursion)
            if isinstance(completion, GeneratorOutput):
                return completion
            
            # Check if it's a ChatCompletion object (non-streaming response)
            if hasattr(completion, 'choices') and hasattr(completion, 'usage'):
                # ALWAYS extract the string content directly
                try:
                    # Direct extraction of message content
                    if (hasattr(completion, 'choices') and 
                        len(completion.choices) > 0 and 
                        hasattr(completion.choices[0], 'message') and 
                        hasattr(completion.choices[0].message, 'content')):
                        
                        content = completion.choices[0].message.content
                        if isinstance(content, str):
                            parsed_data = content
                        else:
                            parsed_data = str(content)
                    else:
                        # Fallback: convert entire completion to string
                        parsed_data = str(completion)
                        
                except Exception as e:
                    # Ultimate fallback
                    parsed_data = str(completion)
                
                return GeneratorOutput(
                    data=parsed_data,
                    usage=CompletionUsage(
                        completion_tokens=completion.usage.completion_tokens,
                        prompt_tokens=completion.usage.prompt_tokens,
                        total_tokens=completion.usage.total_tokens,
                    ),
                    raw_response=str(completion),
                )
            else:
                # Handle streaming response - collect all content parts into a single string
                content_parts = []
                usage_info = None
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        content_parts.append(chunk.choices[0].delta.content)
                    # Try to get usage info from the last chunk
                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage_info = chunk.usage
                
                # Join all content parts into a single string
                full_content = ''.join(content_parts)
                
                # Create usage object
                usage = None
                if usage_info:
                    usage = CompletionUsage(
                        completion_tokens=usage_info.completion_tokens,
                        prompt_tokens=usage_info.prompt_tokens,
                        total_tokens=usage_info.total_tokens,
                    )
                
                return GeneratorOutput(
                    data=full_content,
                    usage=usage,
                    raw_response="streaming"
                )
        except Exception as e:
            log.error(f"Error parsing completion: {e}")
            raise

    def track_completion_usage(
        self,
        completion: Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]],
    ) -> CompletionUsage:
        """Track the completion usage."""
        if isinstance(completion, ChatCompletion):
            return CompletionUsage(
                completion_tokens=completion.usage.completion_tokens,
                prompt_tokens=completion.usage.prompt_tokens,
                total_tokens=completion.usage.total_tokens,
            )
        else:
            # For streaming, we can't track usage accurately
            return CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)

    def parse_embedding_response(
        self, response: CreateEmbeddingResponse
    ) -> EmbedderOutput:
        """Parse the embedding response to a EmbedderOutput."""
        try:
            result = parse_embedding_response(response)
            if result.data:
                log.info(f"🔍 Number of embeddings: {len(result.data)}")
                if len(result.data) > 0:
                    log.info(f"🔍 First embedding length: {len(result.data[0].embedding) if hasattr(result.data[0], 'embedding') else 'N/A'}")
            else:
                log.warning(f"🔍 No embedding data found in result")
            return result
        except Exception as e:
            log.error(f"🔍 Error parsing MoonShot embedding response: {e}")
            log.error(f"🔍 Raw response details: {repr(response)}")
            return EmbedderOutput(data=[], error=str(e), raw_response=response)

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """Convert inputs to API kwargs."""
        final_model_kwargs = model_kwargs.copy()
        
        if model_type == ModelType.LLM:
            messages = []
            if isinstance(input, str):
                messages = [{"role": "user", "content": input}]
            elif isinstance(input, list):
                messages = input
            else:
                raise ValueError(f"Unsupported input type: {type(input)}")
            
            api_kwargs = {
                "messages": messages,
                **final_model_kwargs
            }
            
            return api_kwargs
            
        elif model_type == ModelType.EMBEDDER:
            # Convert Documents to text strings for embedding
            processed_input = input
            if isinstance(input, list):
                # Extract text from Document objects
                processed_input = []
                for item in input:
                    if hasattr(item, 'text'):
                        # It's a Document object, extract text
                        processed_input.append(item.text)
                    elif isinstance(item, str):
                        # It's already a string
                        processed_input.append(item)
                    else:
                        # Try to convert to string
                        processed_input.append(str(item))
            elif hasattr(input, 'text'):
                # Single Document object
                processed_input = input.text
            elif isinstance(input, str):
                # Single string
                processed_input = input
            else:
                # Convert to string as fallback
                processed_input = str(input)
            
            api_kwargs = {
                "input": processed_input,
                **final_model_kwargs
            }
            
            return api_kwargs
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
            BadRequestError,
        ),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """Call the Moonshot API."""
        if model_type == ModelType.LLM:
            completion = self.sync_client.chat.completions.create(**api_kwargs)
            
            if api_kwargs.get("stream", False):
                return handle_streaming_response(completion)
            else:
                return self.parse_chat_completion(completion)
        elif model_type == ModelType.EMBEDDER:
            # Extract input texts from api_kwargs
            texts = api_kwargs.get("input", [])
            
            if not texts:
                log.warning("😭 No input texts provided")
                return EmbedderOutput(data=[], error="No input texts provided", raw_response=None)
            
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
            
            # Filter out empty or None texts - following HuggingFace client pattern
            valid_texts = []
            valid_indices = []
            for i, text in enumerate(texts):
                if text and isinstance(text, str) and text.strip():
                    valid_texts.append(text)
                    valid_indices.append(i)
                else:
                    log.warning(f"🔍 Skipping empty or invalid text at index {i}: type={type(text)}, length={len(text) if hasattr(text, '__len__') else 'N/A'}, repr={repr(text)[:100]}")
            
            if not valid_texts:
                log.error("😭 No valid texts found after filtering")
                return EmbedderOutput(data=[], error="No valid texts found after filtering", raw_response=None)
            
            if len(valid_texts) != len(texts):
                filtered_count = len(texts) - len(valid_texts)
                log.warning(f"🔍 Filtered out {filtered_count} empty/invalid texts out of {len(texts)} total texts")
            
            # Create modified api_kwargs with only valid texts
            filtered_api_kwargs = api_kwargs.copy()
            filtered_api_kwargs["input"] = valid_texts
            
            log.info(f"🔍 MoonShot embedding API call with {len(valid_texts)} valid texts out of {len(texts)} total")
            
            try:
                response = self.sync_client.embeddings.create(**filtered_api_kwargs)
                log.info(f"🔍 MoonShot API call successful, response type: {type(response)}")
                result = self.parse_embedding_response(response)
                
                # If we filtered texts, we need to create embeddings for the original indices
                if len(valid_texts) != len(texts):
                    log.info(f"🔍 Creating embeddings for {len(texts)} original positions")
                    
                    # Get the correct embedding dimension from the first valid embedding
                    embedding_dim = None
                    if result.data and len(result.data) > 0 and hasattr(result.data[0], 'embedding'):
                        embedding_dim = len(result.data[0].embedding)
                        log.info(f"🔍 Using embedding dimension: {embedding_dim}")
                    
                    final_data = []
                    valid_idx = 0
                    for i in range(len(texts)):
                        if i in valid_indices:
                            # Use the embedding from valid texts
                            final_data.append(result.data[valid_idx])
                            valid_idx += 1
                        else:
                            # Create zero embedding for filtered texts with correct dimension
                            log.warning(f"🔍 Creating zero embedding for filtered text at index {i}")
                            final_data.append(Embedding(
                                embedding=[0.0] * embedding_dim,
                                index=i
                            ))
                    
                    result = EmbedderOutput(
                        data=final_data,
                        error=None,
                        raw_response=result.raw_response
                    )
                
                return result
                
            except Exception as e:
                log.error(f"🔍 MoonShot API call failed: {e}")
                return EmbedderOutput(data=[], error=str(e), raw_response=None)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
            BadRequestError,
        ),
        max_time=5,
    )
    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        """Async call to the Moonshot API."""
        if not self.async_client:
            self.async_client = self.init_async_client()

        if model_type == ModelType.LLM:
            completion = await self.async_client.chat.completions.create(**api_kwargs)

            if api_kwargs.get("stream", False):
                return handle_streaming_response(completion)
            else:
                return self.parse_chat_completion(completion)
        elif model_type == ModelType.EMBEDDER:
            # Extract input texts from api_kwargs
            texts = api_kwargs.get("input", [])
            
            if not texts:
                log.warning("😭 No input texts provided")
                return EmbedderOutput(data=[], error="No input texts provided", raw_response=None)
            
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
            
            # Filter out empty or None texts - following HuggingFace client pattern
            valid_texts = []
            valid_indices = []
            for i, text in enumerate(texts):
                if text and isinstance(text, str) and text.strip():
                    valid_texts.append(text)
                    valid_indices.append(i)
                else:
                    log.warning(f"🔍 Skipping empty or invalid text at index {i}: type={type(text)}, length={len(text) if hasattr(text, '__len__') else 'N/A'}, repr={repr(text)[:100]}")
            
            if not valid_texts:
                log.error("😭 No valid texts found after filtering")
                return EmbedderOutput(data=[], error="No valid texts found after filtering", raw_response=None)
            
            if len(valid_texts) != len(texts):
                filtered_count = len(texts) - len(valid_texts)
                log.warning(f"🔍 Filtered out {filtered_count} empty/invalid texts out of {len(texts)} total texts")
            
            # Create modified api_kwargs with only valid texts
            filtered_api_kwargs = api_kwargs.copy()
            filtered_api_kwargs["input"] = valid_texts
            
            log.info(f"🔍 MoonShot async embedding API call with {len(valid_texts)} valid texts out of {len(texts)} total")
            
            try:
                response = await self.async_client.embeddings.create(**filtered_api_kwargs)
                log.info(f"🔍 MoonShot async API call successful, response type: {type(response)}")
                result = self.parse_embedding_response(response)
                
                # If we filtered texts, we need to create embeddings for the original indices
                if len(valid_texts) != len(texts):
                    log.info(f"🔍 Creating embeddings for {len(texts)} original positions")
                    
                    # Get the correct embedding dimension from the first valid embedding
                    embedding_dim = 256  # Default fallback based on config
                    if result.data and len(result.data) > 0 and hasattr(result.data[0], 'embedding'):
                        embedding_dim = len(result.data[0].embedding)
                        log.info(f"🔍 Using embedding dimension: {embedding_dim}")
                    
                    final_data = []
                    valid_idx = 0
                    for i in range(len(texts)):
                        if i in valid_indices:
                            # Use the embedding from valid texts
                            final_data.append(result.data[valid_idx])
                            valid_idx += 1
                        else:
                            # Create zero embedding for filtered texts with correct dimension
                            log.warning(f"🔍 Creating zero embedding for filtered text at index {i}")
                            final_data.append(Embedding(
                                embedding=[0.0] * embedding_dim,
                                index=i
                            ))
                    
                    result = EmbedderOutput(
                        data=final_data,
                        error=None,
                        raw_response=result.raw_response
                    )
                
                return result
                
            except Exception as e:
                log.error(f"🔍 MoonShot async API call failed: {e}")
                return EmbedderOutput(data=[], error=str(e), raw_response=None)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create an instance from a dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "api_key": self._api_key,
            "base_url": self.base_url,
            "input_type": self._input_type,
        }

    def __getstate__(self):
        """
        Customize serialization to exclude non-picklable client objects.
        This method is called by pickle when saving the object's state.
        """
        state = self.__dict__.copy()
        # Remove the unpicklable client instances
        if 'sync_client' in state:
            del state['sync_client']
        if 'async_client' in state:
            del state['async_client']
        return state

    def __setstate__(self, state):
        """
        Customize deserialization to re-create the client objects.
        This method is called by pickle when loading the object's state.
        """
        self.__dict__.update(state)
        # Re-initialize the clients after unpickling
        self.sync_client = self.init_sync_client()
        self.async_client = None  # It will be lazily initialized when acall is used