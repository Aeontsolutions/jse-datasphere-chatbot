"""
Centralized configuration management using Pydantic Settings.

This module provides type-safe, validated configuration for the entire application.
All configuration values are loaded from environment variables with proper validation.
"""

import json
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AWSConfig(BaseSettings):
    """AWS-specific configuration."""

    access_key_id: str = Field(..., description="AWS Access Key ID")
    secret_access_key: str = Field(..., description="AWS Secret Access Key")
    region: str = Field(default="us-east-1", alias="AWS_DEFAULT_REGION", description="AWS Region")
    s3_bucket: str = Field(
        ..., alias="DOCUMENT_METADATA_S3_BUCKET", description="S3 bucket for documents"
    )

    model_config = SettingsConfigDict(
        env_prefix="AWS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
    )


class GoogleCloudConfig(BaseSettings):
    """Google Cloud Platform configuration."""

    project_id: str = Field(..., description="GCP Project ID")
    service_account_info: str = Field(..., description="GCP Service Account JSON as string")
    api_key: str = Field(..., alias="GOOGLE_API_KEY", description="Google API Key for Gemini")
    location: str = Field(default="us-central1", description="GCP region")

    model_config = SettingsConfigDict(
        env_prefix="GCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
    )

    @field_validator("service_account_info")
    @classmethod
    def validate_service_account_json(cls, v: str) -> str:
        """Validate that service account info is valid JSON."""
        try:
            json.loads(v)
            return v
        except json.JSONDecodeError as e:
            raise ValueError(f"GCP_SERVICE_ACCOUNT_INFO must be valid JSON: {e}")

    @property
    def service_account_dict(self) -> dict:
        """Parse service account info as dictionary."""
        return json.loads(self.service_account_info)


class BigQueryConfig(BaseSettings):
    """BigQuery-specific configuration."""

    dataset: str = Field(..., description="BigQuery dataset name")
    table: str = Field(..., description="BigQuery table name")
    location: str = Field(default="US", description="BigQuery location")

    model_config = SettingsConfigDict(
        env_prefix="BIGQUERY_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
    )


class RedisConfig(BaseSettings):
    """Redis configuration for job storage."""

    url: Optional[str] = Field(
        default=None,
        alias="REDIS_URL",
        description="Redis connection URL (environment variable: REDIS_URL)",
    )
    ttl_seconds: int = Field(default=900, description="TTL for job data in seconds")
    max_progress_history: int = Field(default=50, description="Max progress updates to retain")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
    )


class S3DownloadConfig(BaseSettings):
    """S3 download configuration and retry settings."""

    max_retries: int = Field(default=2, description="Maximum retry attempts for S3 downloads")
    retry_delay: float = Field(default=0.5, description="Initial retry delay in seconds")
    max_retry_delay: float = Field(default=10.0, description="Maximum retry delay in seconds")
    timeout: float = Field(default=300.0, description="Download timeout in seconds")
    chunk_size: int = Field(default=8192, description="Chunk size for streaming downloads")
    concurrent_downloads: int = Field(default=5, description="Max concurrent downloads")

    model_config = SettingsConfigDict(
        env_prefix="S3_DOWNLOAD_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
    )


class GeminiAIConfig(BaseSettings):
    """Gemini AI model configuration."""

    model_name: str = Field(default="gemini-2.0-flash-exp", description="Gemini model name")
    temperature: float = Field(default=0.7, description="Model temperature")
    max_output_tokens: int = Field(default=2048, description="Maximum output tokens")
    force_llm_fallback: bool = Field(
        default=True,
        alias="FORCE_LLM_FALLBACK",
        description="Force LLM fallback for document selection",
    )

    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
    )


class AsyncJobConfig(BaseSettings):
    """Async job processing configuration."""

    enabled: bool = Field(default=True, alias="ASYNC_JOB_MODE", description="Enable async job mode")
    ttl_seconds: int = Field(
        default=900, alias="ASYNC_JOB_TTL_SECONDS", description="Job TTL in seconds"
    )
    max_progress_history: int = Field(
        default=50, alias="ASYNC_JOB_MAX_PROGRESS_HISTORY", description="Max progress updates"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
    )


class AppConfig(BaseSettings):
    """
    Main application configuration.

    Aggregates all sub-configurations into a single, type-safe settings object.
    Configuration is loaded from environment variables with validation.
    """

    # Application metadata
    app_name: str = Field(default="jse-datasphere-chatbot", description="Application name")
    log_level: str = Field(default="INFO", description="Logging level")
    metadata_key: str = Field(
        default="metadata_2025-11-26.json", description="S3 key for metadata file"
    )

    # Sub-configurations (loaded from environment)
    aws: AWSConfig = Field(default_factory=AWSConfig)
    gcp: GoogleCloudConfig = Field(default_factory=GoogleCloudConfig)
    bigquery: BigQueryConfig = Field(default_factory=BigQueryConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    s3_download: S3DownloadConfig = Field(default_factory=S3DownloadConfig)
    gemini: GeminiAIConfig = Field(default_factory=GeminiAIConfig)
    async_job: AsyncJobConfig = Field(default_factory=AsyncJobConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the standard Python logging levels."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}, got: {v}")
        return v_upper


# Singleton instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Get the singleton configuration instance.

    This ensures configuration is loaded only once and reused throughout the application.
    Thread-safe lazy initialization.

    Returns:
        AppConfig: The application configuration instance

    Example:
        >>> from app.config import get_config
        >>> config = get_config()
        >>> print(config.aws.region)
        'us-east-1'
    """
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reload_config() -> AppConfig:
    """
    Force reload configuration from environment variables.

    Useful for testing or when environment variables change at runtime.

    Returns:
        AppConfig: The newly loaded configuration instance
    """
    global _config
    _config = AppConfig()
    return _config


# Convenience function for compatibility with existing code
def init_config() -> AppConfig:
    """
    Initialize configuration (alias for get_config).

    Returns:
        AppConfig: The application configuration instance
    """
    return get_config()
