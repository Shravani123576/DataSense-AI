"""
DataSense AI — Central Configuration Settings (Phase 4 / Graph AI Analyst Agent)
Centralizes model parameters, timeouts, step limits, row limits, and environment variables.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Ensure .env is loaded from backend directory or workspace root
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    load_dotenv()


class Settings(BaseModel):
    # Model Configuration
    model_name: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "google:gemini-3.5-flash")
    )
    api_key_env_var: str = "GEMINI_API_KEY"

    # Execution Limits & Timeouts
    max_tool_steps: int = Field(default_factory=lambda: int(os.getenv("MAX_TOOL_STEPS", "6")))
    per_tool_timeout_seconds: float = Field(default_factory=lambda: float(os.getenv("TOOL_TIMEOUT_SECONDS", "30.0")))
    max_rows_returned_to_llm: int = Field(default_factory=lambda: int(os.getenv("MAX_ROWS_RETURNED", "100")))
    upload_max_bytes: int = Field(default_factory=lambda: int(os.getenv("UPLOAD_MAX_BYTES", str(100 * 1024 * 1024))))
    request_timeout_seconds: float = Field(default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60.0")))

    # Grounding Check
    strict_grounding_check: bool = Field(default_factory=lambda: os.getenv("STRICT_GROUNDING", "true").lower() == "true")

    def get_api_key(self, explicit_key: Optional[str] = None) -> Optional[str]:
        """Resolve API key strictly on backend from environment or explicit override."""
        if explicit_key and explicit_key.strip():
            return explicit_key.strip()
        env_val = os.getenv(self.api_key_env_var, "").strip()
        return env_val if env_val else None

    def is_llm_configured(self, explicit_key: Optional[str] = None) -> bool:
        """Check if an API key is available."""
        key = self.get_api_key(explicit_key)
        return bool(key and len(key) > 5)

    def get_configured_model_name(self) -> str:
        """Return the configured model name string."""
        return self.model_name


# Singleton settings instance
settings = Settings()
