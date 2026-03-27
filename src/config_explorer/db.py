"""
Mocks DB storing info about common accelerators used for LLM serving and inference
"""
import json
import os

gpu_specs = {}

_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_dir, "db.json")) as f:
    gpu_specs = json.load(f)
