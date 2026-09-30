"""Demo script for ascendforge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Demo pipeline parallelism (1F1B schedule)."""
from ascendforge.train.pipeline import PipelineParallel
pp = PipelineParallel(num_stages=4, num_microbatches=8)
print("schedule length:", len(pp.schedule()))
print("bubble ratio:", pp.bubble_ratio())
