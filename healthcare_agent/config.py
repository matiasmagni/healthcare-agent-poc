"""
Configuration Module for Healthcare Agent.

This module handles all environment-based configuration, default values,
and application constants. It follows the principle of externalizing
configuration from code to support different environments (dev, staging, prod).

All configuration values are loaded from environment variables with sensible
fallbacks to ensure the agent can run in a local development environment.
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file if present
# This allows for local development without hardcoding sensitive values
load_dotenv()


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Database configuration settings for PostgreSQL connection.
    
    Attributes:
        database_url (str): Full SQLAlchemy connection URL for PostgreSQL.
                           Format: postgresql://user:password@host:port/database
    """
    database_url: str


@dataclass(frozen=True)
class LLMConfig:
    """
    LLM configuration settings for OpenAI-compatible endpoint (Unsloth).
    
    Attributes:
        base_url (str): Base URL for the OpenAI-compatible API endpoint.
        api_key (str): API key for authentication to the LLM endpoint.
        model (str): Model identifier to use for inference.
        temperature (float): Sampling temperature for response generation (0.0 = deterministic).
        max_tokens (Optional[int]): Maximum tokens to generate (None = model default).
    """
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None


@dataclass(frozen=True)
class AgentConfig:
    """
    Agent runtime configuration settings.
    
    Attributes:
        max_iterations (int): Maximum number of reasoning/tool-calling iterations.
        verbose (bool): Enable detailed logging of agent thought process.
        system_prompt (str): System instructions guiding agent behavior.
    """
    max_iterations: int
    verbose: bool
    system_prompt: str


@dataclass(frozen=True)
class AppConfig:
    """
    Root application configuration container.
    
    Aggregates all configuration domains for centralized access.
    
    Attributes:
        database (DatabaseConfig): Database connection configuration.
        llm (LLMConfig): LLM endpoint and model configuration.
        agent (AgentConfig): Agent behavior configuration.
    """
    database: DatabaseConfig
    llm: LLMConfig
    agent: AgentConfig


def load_config() -> AppConfig:
    """
    Load and validate all application configuration from environment variables.
    
    This function implements defensive configuration loading with explicit
    fallbacks to ensure the application can start in development environments.
    All sensitive values should be provided via environment variables in production.
    
    Returns:
        AppConfig: Fully initialized configuration object with all settings.
    """
    # -------------------- Database Configuration --------------------
    database_url: str = os.getenv("DATABASE_URL", "")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required")
    
    # -------------------- LLM Configuration --------------------
    unsloth_base_url: str = os.getenv("UNSLOTH_BASE_URL", "")
    if not unsloth_base_url:
        raise ValueError("UNSLOTH_BASE_URL environment variable is required")
    
    unsloth_api_key: str = os.getenv("UNSLOTH_API_KEY", "")
    if not unsloth_api_key:
        raise ValueError("UNSLOTH_API_KEY environment variable is required")
    
    unsloth_model: str = os.getenv("UNSLOTH_MODEL", "")
    if not unsloth_model:
        raise ValueError("UNSLOTH_MODEL environment variable is required")
    
    agent_temperature: float = float(os.getenv("AGENT_TEMPERATURE", "0.0"))
    
    # -------------------- Agent Configuration --------------------
    agent_max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
    agent_verbose: bool = os.getenv("AGENT_VERBOSE", "true").lower() == "true"
    
    # System prompt defining agent role, capabilities, and constraints
    system_prompt: str = (
        "You are a Senior Healthcare Data Agent with specialized access to a PostgreSQL database "
        "containing clean medical records. Your primary responsibility is to assist authorized users "
        "with querying and analyzing patient information safely and accurately.\n\n"
        "CORE CAPABILITIES:\n"
        "- Search for patients by first and last name\n"
        "- Retrieve detailed billing summaries including costs, hospital, insurance, and length of stay\n"
        "- Query patient populations by medical condition with configurable result limits\n\n"
        "OPERATIONAL GUIDELINES:\n"
        "1. Always use the provided tools to retrieve data. Never hallucinate patient records.\n"
        "2. Be precise and factual in all responses. Present data in a clear, structured format.\n"
        "3. When searching by name, use both first and last name for maximum accuracy.\n"
        "4. For billing summaries, always calculate length of stay from admission and discharge dates.\n"
        "5. Respect data privacy - only return information requested and never expose unrelated records.\n"
        "6. If a query returns no results, clearly communicate this to the user.\n"
        "7. Handle errors gracefully and explain issues in user-friendly language.\n"
        "8. When multiple patients match a name search, present all matches with distinguishing details.\n\n"
        "You have access to three specialized tools. Use them appropriately based on the user's request."
    )
    
    # -------------------- Assemble Configuration --------------------
    config = AppConfig(
        database=DatabaseConfig(database_url=database_url),
        llm=LLMConfig(
            base_url=unsloth_base_url,
            api_key=unsloth_api_key,
            model=unsloth_model,
            temperature=agent_temperature
        ),
        agent=AgentConfig(
            max_iterations=agent_max_iterations,
            verbose=agent_verbose,
            system_prompt=system_prompt
        )
    )
    
    return config


# Global configuration instance for application-wide access
# Loaded once at module import time
APP_CONFIG: AppConfig = load_config()
