"""
config.py — load local credentials from .env into the environment.

Dependency-free (no python-dotenv). Splits each line on the FIRST '=' so base64
bearer tokens containing '=' survive. Uses setdefault so a real ambient AWS
profile/env wins over .env if present. The .env itself is gitignored.
"""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# The Claude model verified available on this account (base id needs the "us."
# cross-region inference-profile prefix). Haiku 4.5 = cheap + strong grounding.
BEDROCK_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def load_env(path: str | Path | None = None) -> None:
    p = Path(path) if path else _ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())
