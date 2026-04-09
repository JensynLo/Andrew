import json


def load_config(cfg_path: str):
    with open(cfg_path, "r") as f:
        return json.load(f)
