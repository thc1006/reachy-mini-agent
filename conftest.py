"""pytest discovery: make src/ importable so tests can `from streaming_tts import ...`."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
