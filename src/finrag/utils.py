import os


def get_env_var(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"environment variable: {name} is required but missing.")
    return value
