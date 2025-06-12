"""
LLM Configuration for NavTech Resume Parser
Defines available providers and their configurations
"""

from typing import Dict, Any, List


class LLMConfig:
    """Configuration class for LLM providers"""
    
    # Default configurations for each provider
    GEMINI_CONFIG = {
        "model": "gemini-1.5-pro",
        "temperature": 0.1,
        "max_tokens": 4000
    }
    
    OPENAI_CONFIG = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.1,
        "max_tokens": 4000
    }
    
    OPENROUTER_CONFIG = {
        "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
        "temperature": 0.1,
        "max_tokens": 4000
    }
    
    TRANSFORMER_CONFIG = {
        "model": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "device": "auto"
    }


# Available LLM providers for the application
AVAILABLE_PROVIDERS = [
    {
        "name": "gemini",
        "display_name": "Google Gemini",
        "description": "Google's Gemini 1.5 Pro model for high-quality resume parsing",
        "requires_api_key": True,
        "config": LLMConfig.GEMINI_CONFIG
    },
    {
        "name": "openai",
        "display_name": "OpenAI GPT",
        "description": "OpenAI's GPT-3.5-turbo for reliable resume parsing",
        "requires_api_key": True,
        "config": LLMConfig.OPENAI_CONFIG
    },
    {
        "name": "openrouter",
        "display_name": "OpenRouter (DeepSeek R1)",
        "description": "Free DeepSeek R1 model via OpenRouter - excellent accuracy",
        "requires_api_key": True,
        "config": LLMConfig.OPENROUTER_CONFIG
    },
    {
        "name": "smart_transformer",
        "display_name": "Enhanced Smart PDF + Transformer",
        "description": "Local transformer model with enhanced PDF processing (85.7% accuracy)",
        "requires_api_key": False,
        "config": LLMConfig.TRANSFORMER_CONFIG
    },
    {
        "name": "layoutlm_transformer",
        "display_name": "Improved LayoutLM + Transformer",
        "description": "Advanced LayoutLM-based transformer model (85.7% accuracy)",
        "requires_api_key": False,
        "config": LLMConfig.TRANSFORMER_CONFIG
    }
]


def get_provider_config(provider_name: str) -> Dict[str, Any]:
    """Get configuration for a specific provider"""
    for provider in AVAILABLE_PROVIDERS:
        if provider["name"] == provider_name:
            return provider["config"]
    
    raise ValueError(f"Unknown provider: {provider_name}")


def get_provider_info(provider_name: str) -> Dict[str, Any]:
    """Get full information for a specific provider"""
    for provider in AVAILABLE_PROVIDERS:
        if provider["name"] == provider_name:
            return provider
    
    raise ValueError(f"Unknown provider: {provider_name}")


def list_available_providers() -> List[str]:
    """Get list of available provider names"""
    return [provider["name"] for provider in AVAILABLE_PROVIDERS]
