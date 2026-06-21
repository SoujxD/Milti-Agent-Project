"""Shared pytest configuration.

Ensures the repository root is importable and forces the project into its
deterministic offline mock mode during tests by clearing any API keys. This
makes the suite reproducible and lets it run in CI without secrets.
"""

import os
import sys
from pathlib import Path

# Make the repo root importable (so `import agents...` / `import evaluation...` work).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force offline mock paths: no LLM calls, deterministic behavior.
for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
    os.environ.pop(key, None)
