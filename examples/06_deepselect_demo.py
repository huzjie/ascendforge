"""Demo script for ascendforge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Demo DeepSelect: dedup + quality + curriculum."""
from ascendforge.data.synth import SyntheticCorpus
from ascendforge.data.select import DeepSelectPipeline

docs = SyntheticCorpus(num_docs=200).generate()
pipe = DeepSelectPipeline(quality_min=0.0)
kept = pipe.run(docs)
print(f"before={len(docs)} after={len(kept)} removed={len(docs) - len(kept)}")
