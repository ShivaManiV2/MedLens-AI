"""Quick test: bypass SSE streaming and call generation directly."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from retrieval import RetrievalPipeline
from generation import GenerationPipeline

print("Loading pipelines...")
ret = RetrievalPipeline()
gen = GenerationPipeline()
print("Pipelines loaded.")

q = "What does research say about Vitamin D and bone density?"
print(f"\nQuery: {q}")

chunks = ret.retrieve_and_rerank(q)
print(f"Retrieved {len(chunks)} chunks.")

print("\nStreaming answer:")
try:
    for token in gen.stream_answer(q, chunks):
        print(token, end="", flush=True)
    print("\n\n--- DONE ---")
except Exception as e:
    print(f"\n\nERROR during streaming: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
