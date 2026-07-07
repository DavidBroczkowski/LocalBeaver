"""CoNLL Experiment - model must correctly classify proper nouns via named entity recoginition"""
import json
import torch
import numpy as np
from pathlib import Path

DATASET_NAME = "conll"

_DATA_DIR = Path(__file__).parent / "data"
_DEFAULT_DATASET_PATH = str(_DATA_DIR / "train.json")

def load_input_rows():
    """
    Returns the data in the json file for the dataset

    Output:
        - a dictionary containing the input and appropriate tags
    """
    with open(_DEFAULT_DATASET_PATH, 'r') as file:
        data = json.load(file)
        return data

def load_prompts(start_idx: int = 0, end_idx: int = -1, **kwargs) -> list[dict]:
    data = load_input_rows()
    inputs = data["inputs"]
    tags = data["tags"]

    instances = []
    end = end_idx if end_idx != -1 else len(inputs)
    for i in range(start_idx, end):
        instances.append(
            {
                "prompt": inputs[i],
                "inputs": inputs[i],
                "tags": tags[i]
            }
        )

    return instances

def constraint_fn(instance: dict, sequence: str) -> bool:
    """True = acceptable, False = violation."""
    # print("RUNNING CONSTRAINT_FN")
    # print(f"[DEBUG] sequence after split: {sequence.split()}")
    # print(f"[DEBUG] tags: {instance['tags']}")
    # print(f"[DEBUG] equality: {sequence.split() == instance['tags']}")
    return sequence.split() == instance["tags"]

def check_call_fn(instance, decoded_sequences, token_lists):
    """Optional fast pre-filter — skip expensive checks on short prefixes."""
    return np.zeros(len(decoded_sequences), dtype=bool)

def instance_context_fn(instance: dict) -> str:
    """Cache key for this instance's constraint context."""
    return ",".join(instance["tags"])

if __name__ == "__main__":
    import argparse
    import beaver

    parser = argparse.ArgumentParser(description="Run CoNLL-2003 experiment.")
    parser.add_argument("--model", required=True) # must be a path to the model 
    parser.add_argument("--log_dir", default="beaver_logs")
    parser.add_argument("--glove_embed", default=1)
    args, _ = parser.parse_known_args()

    beaver.run(
        prompts=load_prompts(),
        constraint_fn=constraint_fn,
        check_call_fn=check_call_fn,
        cache=True,
        cache_dataset_name=DATASET_NAME,
        instance_context_fn=instance_context_fn,
        model=torch.load(args.model, weights_only=False).eval(),
        log_dir=args.log_dir,
        glove_embed=args.glove_embed
    )
