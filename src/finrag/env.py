from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_DOTENV_LOADED = False


def load_project_dotenv(*, override: bool = False) -> bool:
    """
    Load environment variables from the project root `.env` once per process.

    This is intentionally side-effecting to make CLIs/scripts work without
    requiring callers to remember to call `load_dotenv()` themselves.
    """

    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return False

    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path, override=override)
    else:
        load_dotenv(override=override)

    _DOTENV_LOADED = True
    return True
