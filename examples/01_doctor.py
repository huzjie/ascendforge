"""Demo script for ascendforge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ascendforge.cli import main

if __name__ == "__main__":
    main(["doctor"])
