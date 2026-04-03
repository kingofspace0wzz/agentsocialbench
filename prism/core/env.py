# prism/core/env.py
"""Environment loading and API key management. Forked from MAGPIE."""
import os


def load_env_file(env_path: str = None):
    """Load environment variables from .env file."""
    if env_path is None:
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


def get_gemini_api_keys() -> list[str]:
    """Get Gemini API keys from environment variables."""
    keys = []
    for i in range(1, 6):
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key:
            keys.append(key)
    primary_key = os.getenv("GEMINI_API_KEY")
    if primary_key and primary_key not in keys:
        keys.append(primary_key)
    return keys
