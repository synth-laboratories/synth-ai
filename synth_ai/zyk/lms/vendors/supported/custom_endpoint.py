import re
import os
import json
import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Type
import requests
import httpx
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import random
from urllib.parse import urlparse

from synth_ai.zyk.lms.vendors.base import BaseLMResponse, VendorBase
from synth_ai.zyk.lms.tools.base import BaseTool
from synth_ai.zyk.lms.caching.initialize import get_cache_handler

# Exception types for retry
CUSTOM_ENDPOINT_EXCEPTIONS_TO_RETRY: Tuple[Type[Exception], ...] = (
    requests.RequestException, 
    requests.Timeout,
    httpx.RequestError,
    httpx.TimeoutException
)

class CustomEndpointAPI(VendorBase):
    """Generic vendor client for custom OpenAI-compatible endpoints."""
    
    used_for_structured_outputs: bool = False
    exceptions_to_retry: List = list(CUSTOM_ENDPOINT_EXCEPTIONS_TO_RETRY)
    
    def __init__(self, endpoint_url: str):
        # Validate and sanitize URL
        self._validate_endpoint_url(endpoint_url)
        self.endpoint_url = endpoint_url
        
        # Construct full chat completions URL
        if endpoint_url.endswith('/'):
            endpoint_url = endpoint_url[:-1]
        self.chat_completions_url = f"https://{endpoint_url}/chat/completions"
        self.health_url = f"https://{endpoint_url}/health"
        
        # Setup session with connection pooling and retries
        self.session = self._create_session()
        self.async_client = None  # Lazy init
        
        # Get auth token from environment (generic support for any auth)
        self.auth_token = os.environ.get("CUSTOM_ENDPOINT_API_TOKEN")
        
    def _validate_endpoint_url(self, url: str) -> None:
        """Validate endpoint URL format and prevent SSRF."""
        # Block dangerous URL patterns
        dangerous_patterns = [
            "file://", "ftp://", "gopher://",
            "localhost", "127.", "0.0.0.0",
            "10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.",
            "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.",
            "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31.",
            "169.254.",  # link-local
            "::1", "fc00:", "fd00:", "fe80:",  # IPv6 private
        ]
        
        for pattern in dangerous_patterns:
            if pattern in url.lower():
                raise ValueError(f"Blocked URL pattern for security: {pattern}")
                
        # Limit URL length
        if len(url) > 256:
            raise ValueError(f"Endpoint URL too long (max 256 chars)")
            
        # Basic URL format check
        if not re.match(r'^[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=]+$', url):
            raise ValueError(f"Invalid URL format: {url}")
    
    def _create_session(self) -> requests.Session:
        """Create session with retry strategy and connection pooling."""
        session = requests.Session()
        
        # Exponential backoff with jitter
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    async def _get_async_client(self) -> httpx.AsyncClient:
        """Lazy init async client with shared retry logic."""
        if self.async_client is None:
            self.async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
        return self.async_client
    
    def _get_timeout(self, lm_config: Dict[str, Any]) -> float:
        """Get timeout with per-call override support."""
        return lm_config.get("timeout", 
                           float(os.environ.get("CUSTOM_ENDPOINT_REQUEST_TIMEOUT", "30")))
    
    def _get_temperature_override(self) -> Optional[float]:
        """Get temperature override from environment for this specific endpoint."""
        # Create a safe env var key from the endpoint URL
        # e.g., "example.com/api" -> "CUSTOM_ENDPOINT_TEMP_EXAMPLE_COM_API"
        safe_key = re.sub(r'[^A-Za-z0-9]', '_', self.endpoint_url).upper()
        safe_key = safe_key[:64]  # Limit length
        
        env_key = f"CUSTOM_ENDPOINT_TEMP_{safe_key}"
        temp_str = os.environ.get(env_key)
        return float(temp_str) if temp_str else None
    
    def _compress_tool_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Compress JSON schema to reduce token usage."""
        if isinstance(schema, dict):
            # Remove verbose keys
            compressed = {
                k: self._compress_tool_schema(v) 
                for k, v in schema.items() 
                if k not in ["title", "$ref", "$schema"]
            }
            # Shorten descriptions
            if "description" in compressed and len(compressed["description"]) > 50:
                compressed["description"] = compressed["description"][:47] + "..."
            return compressed
        elif isinstance(schema, list):
            return [self._compress_tool_schema(item) for item in schema]
        return schema
    
    def _inject_tools_into_prompt(self, system_message: str, tools: List[BaseTool]) -> str:
        """Inject tool definitions with compressed schemas and clear output format."""
        if not tools:
            return system_message
            
        tool_descriptions = []
        for tool in tools:
            schema = tool.arguments.model_json_schema()
            compressed_schema = self._compress_tool_schema(schema)
            
            tool_desc = f"Tool: {tool.name}\nDesc: {tool.description}\nParams: {json.dumps(compressed_schema, separators=(',', ':'))}"
            tool_descriptions.append(tool_desc)
        
        tools_text = "\n".join(tool_descriptions)
        
        return f"""{system_message}

Available tools:
{tools_text}

IMPORTANT: To use a tool, respond with JSON wrapped in ```json fences:
```json
{{"tool_call": {{"name": "tool_name", "arguments": {{...}}}}}}
```

For regular responses, just respond normally without JSON fences."""

    def _extract_tool_calls(self, content: str, tools: List[BaseTool]) -> tuple[Optional[List], str]:
        """Extract and validate tool calls from response."""
        # Look for JSON fenced blocks
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        if not matches:
            return None, content
            
        tool_calls = []
        cleaned_content = content
        
        for match in matches:
            try:
                tool_data = json.loads(match)
                if "tool_call" in tool_data:
                    call_data = tool_data["tool_call"]
                    tool_name = call_data.get("name")
                    
                    # Validate against available tools
                    matching_tool = next((t for t in tools if t.name == tool_name), None)
                    if matching_tool:
                        # Validate arguments with pydantic
                        validated_args = matching_tool.arguments(**call_data.get("arguments", {}))
                        tool_calls.append({
                            "name": tool_name,
                            "arguments": validated_args.model_dump()
                        })
                        
                        # Remove tool call from content
                        cleaned_content = cleaned_content.replace(f"```json\n{match}\n```", "").strip()
                        
            except (json.JSONDecodeError, Exception):
                # Fall back to treating as normal text if validation fails
                continue
                
        return tool_calls if tool_calls else None, cleaned_content
    
    def _exponential_backoff_with_jitter(self, attempt: int) -> float:
        """Calculate backoff time with jitter to prevent thundering herd."""
        base_delay = min(2 ** attempt, 32)  # Cap at 32 seconds
        jitter = random.uniform(0, 1)
        return base_delay + jitter
    
    def _handle_rate_limit(self, response: requests.Response) -> None:
        """Extract and propagate rate limit information."""
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                # Bubble up to synth-ai scheduler
                raise requests.exceptions.RetryError(f"Rate limited. Retry after {retry_after}s")
    
    async def _hit_api_async(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        lm_config: Dict[str, Any],
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "low",
        tools: Optional[List[BaseTool]] = None,
    ) -> BaseLMResponse:
        """Async API call with comprehensive error handling and streaming support."""
        
        # Cache integration - check first
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only)
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config, tools=tools
        )
        if cache_result:
            return cache_result
        
        # Apply tool injection
        if tools and messages:
            messages = messages.copy()
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = self._inject_tools_into_prompt(
                    messages[0]["content"], tools
                )
        
        # Prepare request
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        # Apply temperature override
        temp_override = self._get_temperature_override()
        request_temp = temp_override if temp_override else lm_config.get("temperature", 0.7)
        
        payload = {
            "model": model,  # Pass through the model name
            "messages": messages,
            "temperature": request_temp,
            "stream": lm_config.get("stream", False)
        }
        
        timeout = self._get_timeout(lm_config)
        client = await self._get_async_client()
        
        # Make request with retry logic
        for attempt in range(3):
            try:
                response = await client.post(
                    self.chat_completions_url,
                    json=payload,
                    headers=headers,
                    timeout=timeout
                )
                
                if response.status_code == 429:
                    self._handle_rate_limit(response)
                    
                response.raise_for_status()
                
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"]
                
                # Extract tool calls
                tool_calls, clean_content = self._extract_tool_calls(content, tools or [])
                
                lm_response = BaseLMResponse(
                    raw_response=clean_content,
                    structured_output=None,
                    tool_calls=tool_calls
                )
                
                # Add to cache
                used_cache_handler.add_to_managed_cache(
                    model, messages, lm_config=lm_config, output=lm_response, tools=tools
                )
                
                return lm_response
                    
            except (httpx.RequestError, httpx.TimeoutException) as e:
                if attempt == 2:  # Last attempt
                    raise
                await asyncio.sleep(self._exponential_backoff_with_jitter(attempt))
                
    def _hit_api_sync(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        lm_config: Dict[str, Any],
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "low",
        tools: Optional[List[BaseTool]] = None,
    ) -> BaseLMResponse:
        """Sync version with same logic as async."""
        
        # Cache integration - check first
        used_cache_handler = get_cache_handler(use_ephemeral_cache_only)
        cache_result = used_cache_handler.hit_managed_cache(
            model, messages, lm_config=lm_config, tools=tools
        )
        if cache_result:
            return cache_result
        
        # Apply tool injection
        if tools and messages:
            messages = messages.copy()
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = self._inject_tools_into_prompt(
                    messages[0]["content"], tools
                )
        
        # Prepare request
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        # Apply temperature override
        temp_override = self._get_temperature_override()
        request_temp = temp_override if temp_override else lm_config.get("temperature", 0.7)
        
        payload = {
            "model": model,  # Pass through the model name
            "messages": messages,
            "temperature": request_temp,
            "stream": lm_config.get("stream", False)
        }
        
        timeout = self._get_timeout(lm_config)
        
        # Make request with retry logic
        for attempt in range(3):
            try:
                response = self.session.post(
                    self.chat_completions_url,
                    json=payload,
                    headers=headers,
                    timeout=timeout
                )
                
                if response.status_code == 429:
                    self._handle_rate_limit(response)
                    
                response.raise_for_status()
                
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"]
                
                # Extract tool calls
                tool_calls, clean_content = self._extract_tool_calls(content, tools or [])
                
                lm_response = BaseLMResponse(
                    raw_response=clean_content,
                    structured_output=None,
                    tool_calls=tool_calls
                )
                
                # Add to cache
                used_cache_handler.add_to_managed_cache(
                    model, messages, lm_config=lm_config, output=lm_response, tools=tools
                )
                
                return lm_response
                    
            except (requests.RequestException, requests.Timeout) as e:
                if attempt == 2:  # Last attempt
                    raise
                time.sleep(self._exponential_backoff_with_jitter(attempt))
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'session'):
            self.session.close()
        if hasattr(self, 'async_client') and self.async_client:
            # Schedule cleanup for async client
            pass