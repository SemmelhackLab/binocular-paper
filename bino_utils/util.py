import numpy as np


def load_config():
    from pathlib import Path
    import toml

    config_path = Path(__file__).parent / "config.toml"
    with open(config_path, "r") as f:
        config = toml.load(f)
    return config


def get_bbox(a):
    from itertools import combinations

    def _get_bbox_1d(a):
        return np.array((a.argmax(), len(a) - a[::-1].argmax()))

    a = np.asarray(a).astype(bool)
    return np.array(
        [
            _get_bbox_1d(a.any(i))
            for i in combinations(reversed(range(a.ndim)), a.ndim - 1)
        ]
    )
