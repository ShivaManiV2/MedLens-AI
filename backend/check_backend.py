import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Set up paths
backend_dir = Path(__file__).parent
sys.path.append(str(backend_dir))

print(f"Checking directory: {backend_dir}")
env_path = backend_dir / ".env"
print(f"Checking .env at: {env_path}")
print(f"Exists: {env_path.exists()}")

load_dotenv(dotenv_path=env_path)
key = os.environ.get("GOOGLE_API_KEY")
if key:
    print(f"GOOGLE_API_KEY found: {key[:5]}...{key[-5:]}")
else:
    print("GOOGLE_API_KEY NOT FOUND")

try:
    from src.retrieval import RetrievalPipeline
    from src.generation import GenerationPipeline
    print("Importing pipelines...")
    r = RetrievalPipeline()
    print("RetrievalPipeline initialized.")
    g = GenerationPipeline()
    print("GenerationPipeline initialized.")
    print("SUCCESS: All systems ready.")
except Exception as e:
    print(f"FAILURE: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
