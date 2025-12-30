"""
Backwards-compatible alias for the eval runner.

Prefer calling `scripts/run_eval.py` directly.
"""

from pathlib import Path

from dotenv import load_dotenv

from run_eval import main

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


if __name__ == "__main__":
    main()
