"""
API client implementations for different LLM providers.
"""

import time
import backoff
from abc import ABC, abstractmethod
from typing import Optional


class BaseAPIClient(ABC):
    """Base class for all API clients."""
    
    def __init__(self):
        self.total_cost = 0
        self.error_count = 0
        self.success_count = 0
        self.last_request_time = 0
        self.min_request_interval = 1.5  # Minimum seconds between requests
    
    @abstractmethod
    def send_request(self, model_name: str, prompt: str, max_tokens: int = 150, 
                    temperature: float = 0, **kwargs) -> str:
        """Send a request to the API."""
        pass
    
    def get_total_cost(self) -> float:
        """Get the total accumulated cost of all API calls."""
        return self.total_cost
    
    def get_success_rate(self) -> float:
        """Get the success rate as a percentage."""
        total_requests = self.success_count + self.error_count
        if total_requests == 0:
            return 0
        return (self.success_count / total_requests) * 100
    
    def verify_connection(self) -> tuple[bool, str]:
        """Verify that the client can connect and authenticate."""
        try:
            test_message = "Hello, this is a test message to verify the connection."
            self.send_request(self.deployment_name, test_message, max_tokens=10)
            return True, "Connection successful"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def _enforce_rate_limit(self):
        """Enforce minimum time between requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last_request
            time.sleep(wait_time)


class AzureOpenAIClient(BaseAPIClient):
    """Azure OpenAI API client."""
    
    def __init__(self, api_key: str, endpoint: str, deployment_name: str, 
                 api_version: str = '2024-12-01-preview'):
        super().__init__()
        
        try:
            from openai import AzureOpenAI, RateLimitError, APITimeoutError, OpenAIError
            self.RateLimitError = RateLimitError
            self.APITimeoutError = APITimeoutError
            self.OpenAIError = OpenAIError
        except ImportError:
            raise ImportError("openai package is required for Azure OpenAI client")
        
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self.deployment_name = deployment_name
    
    @backoff.on_exception(
        backoff.expo, 
        (Exception,),  # Will be replaced with specific exceptions in send_request
        max_tries=10,
        max_value=120,
        factor=2,
        jitter=None
    )
    def send_request(self, model_name: str, prompt: str, max_tokens: int = 500, 
                    temperature: float = 0, top_p: float = 1.0, **kwargs) -> str:
        """Send a request to Azure OpenAI API."""
        try:
            self._enforce_rate_limit()
            
            messages = [{"role": "user", "content": prompt}]
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=1,
                timeout=60,
            )
            
            generated_text = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            # Calculate and accumulate cost
            cost = self._calculate_price(prompt_tokens, completion_tokens)
            self.total_cost += cost
            self.success_count += 1
            
            self.last_request_time = time.time()
            
            return generated_text
            
        except Exception as e:
            self.error_count += 1
            
            # Handle rate limits with longer sleep
            if "429" in str(e):
                wait_time = 30
                import re
                match = re.search(r"retry after (\d+) seconds", str(e), re.IGNORECASE)
                if match:
                    suggested_wait = int(match.group(1))
                    wait_time = suggested_wait + 5
                
                print(f"Rate limit exceeded. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            
            raise Exception(f"An error occurred: {str(e)}")
    
    def _calculate_price(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate the price based on token usage and model pricing."""
        model_pricing = {
            'z-gpt-4o-2024-08-0': {'input': 5.00 / 1000, 'output': 15.00 / 1000},
            'z-gpt-4o-mini-2024-07-18': {'input': 0.15 / 1000, 'output': 0.6 / 1000},
            'z-gpt-o1-mini-2024-09-12': {'input': 3 / 1000, 'output': 12 / 1000},
            'z-gpt-o1-preview-2024-09-12': {'input': 15 / 1000, 'output': 60 / 1000},
            'z-gpt-o3-mini-2025-01-31': {'input': 1.10 / 1000, 'output': 4.40 / 1000},
            'default': {'input': 0.0, 'output': 0.0}
        }
        
        pricing = model_pricing.get(self.deployment_name, model_pricing['default'])
        
        input_cost = (prompt_tokens / 1000) * pricing['input']
        output_cost = (completion_tokens / 1000) * pricing['output']
        
        return input_cost + output_cost


class OpenRouterClient(BaseAPIClient):
    """OpenRouter API client."""
    
    def __init__(self, api_key: str, model_name: str, 
                 base_url: str = "https://openrouter.ai/api/v1",
                 site_url: Optional[str] = None, site_name: Optional[str] = None):
        super().__init__()
        
        try:
            from openai import OpenAI, RateLimitError, OpenAIError
            self.RateLimitError = RateLimitError
            self.OpenAIError = OpenAIError
        except ImportError:
            raise ImportError("openai package is required for OpenRouter client")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.deployment_name = model_name
        
        self.extra_headers = {}
        if site_url:
            self.extra_headers["HTTP-Referer"] = site_url
        if site_name:
            self.extra_headers["X-Title"] = site_name
    
    @backoff.on_exception(
        backoff.expo,
        (Exception,),  # Will be replaced with specific exceptions in send_request
        max_tries=10,
        max_value=120,
        factor=2,
        jitter=None
    )
    def send_request(self, model_name: str, prompt: str, max_tokens: int = 150, 
                    temperature: float = 0, **kwargs) -> str:
        """Send a request to OpenRouter API."""
        try:
            self._enforce_rate_limit()
            
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
                extra_headers=self.extra_headers,
                **kwargs
            )
            
            if (not response or not response.choices or
                not response.choices[0].message.content):
                raise ValueError("Invalid response received from API")
            
            generated_text = response.choices[0].message.content.strip()
            self.success_count += 1
            self.last_request_time = time.time()
            
            return generated_text
            
        except Exception as e:
            self.error_count += 1
            
            if "429" in str(e) or "rate limit" in str(e).lower():
                wait_time = 20
                print(f"Rate limit exceeded. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            
            raise e


class DeepSeekClient(BaseAPIClient):
    """DeepSeek API client."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 model_name: str = "deepseek-chat"):
        super().__init__()
        
        try:
            from openai import OpenAI, RateLimitError, OpenAIError
            self.RateLimitError = RateLimitError
            self.OpenAIError = OpenAIError
        except ImportError:
            raise ImportError("openai package is required for DeepSeek client")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.deployment_name = model_name
    
    @backoff.on_exception(
        backoff.expo,
        (Exception,),  # Will be replaced with specific exceptions in send_request
        max_tries=10,
        max_value=120,
        factor=2,
        jitter=None
    )
    def send_request(self, model_name: str, prompt: str, max_tokens: int = 150, 
                    temperature: float = 0, **kwargs) -> str:
        """Send a request to DeepSeek API."""
        try:
            self._enforce_rate_limit()
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
                **kwargs
            )
            
            if (not response or not response.choices or
                not response.choices[0].message.content):
                raise ValueError("Invalid response received from API")
            
            generated_text = response.choices[0].message.content.strip()
            self.success_count += 1
            self.last_request_time = time.time()
            
            return generated_text
            
        except Exception as e:
            self.error_count += 1
            
            if "429" in str(e) or "rate limit" in str(e).lower():
                wait_time = 20
                print(f"Rate limit exceeded. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            
            raise e