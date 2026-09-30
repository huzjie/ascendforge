"""Demo script for ascendforge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Demo MCP handler (single request, without stdio loop)."""
from ascendforge.integrations.mcp import AscendForgeMCPServer
srv = AscendForgeMCPServer()
print(srv.handle("tools/list", {}))
print(srv.handle("tools/call", {"name": "ascendforge_doctor", "arguments": {}}))
