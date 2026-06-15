import os
from dotenv import load_dotenv

# Load environment variables from .env file
# Load environment variables from .env file relative to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=env_path)

class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL")

    @classmethod
    def validate(cls):
        """Validate that all required configuration variables are present."""
        missing = []
        for attr in ["SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY", "DATABASE_URL"]:
            if not getattr(cls, attr):
                missing.append(attr)
        if missing:
            raise ValueError(f"Missing configuration variables in environment: {', '.join(missing)}")
        print("Configuration is valid.")

if __name__ == "__main__":
    try:
        Config.validate()
    except Exception as e:
        print("Validation failed:", e)
