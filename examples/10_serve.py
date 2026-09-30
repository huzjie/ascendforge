"""Demo script for ascendforge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Start the OpenAI-compatible HTTP server."""
from ascendforge.serving.http_server import serve
serve(backend="mock", host="127.0.0.1", port=8000)
