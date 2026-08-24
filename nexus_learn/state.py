import json
from pathlib import Path
import numpy as np
from .model import NumpyRegressor


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        path,
        W1=model.W1,
        b1=model.b1,
        W2=model.W2,
        b2=model.b2,
    )


def load_model(path, seed=17):
    path = Path(path)

    if not path.exists():
        return NumpyRegressor(seed=seed)

    data = np.load(path)

    model = NumpyRegressor(
        seed=seed,
        dim=int(data["W1"].shape[0]),
        hidden=int(data["W1"].shape[1]),
    )

    model.W1 = data["W1"]
    model.b1 = data["b1"]
    model.W2 = data["W2"]
    model.b2 = data["b2"]

    return model


def save_metadata(path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(path):
    path = Path(path)

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
