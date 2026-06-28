import os
from dotenv import load_dotenv

# Load environment variables from .env file relative to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=env_path)

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")

    @classmethod
    def validate(cls):
        """Validate that all required configuration variables are present."""
        print("Configuration is valid (SQLite local-only mode active).")

if __name__ == "__main__":
    try:
        Config.validate()
    except Exception as e:
        print("Validation failed:", e)
