import os

from loguru import logger


def get_env_var(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"environment variable: {name} is required but missing.")
    return value


def seed_everything(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to: {seed}")
