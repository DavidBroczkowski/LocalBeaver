<h1 align="center">
  <img width="60" alt="beaver" src="https://beaverframework.pages.dev/assets/icon.png" />
  &nbsp;LocalBeaver Framework
</h1>

<p align="left">
    🌐&nbsp;<a href="https://beaverframework.pages.dev">Original BEAVER Website</a>
    | 📄&nbsp;<a href="https://arxiv.org/abs/2512.05439">Original BEAVER Paper</a>
    | 💻&nbsp;<a href="https://github.com/uiuc-focal-lab/Beaver">BEAVER GitHub</a>
</p>

<p>
    <a href="https://arxiv.org/abs/2512.05439"><img src="https://img.shields.io/badge/arXiv-2512.05439-b31b1b.svg"></a>
    <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10+-1f425f.svg?color=purple"></a>
    <a href="LICENSE.md"><img alt="BEAVER License" src="https://img.shields.io/badge/License-CC_BY--SA_4.0-blue"></a>
</p>

## ℹ️ About
- **LocalBEAVER** is a modified version of **BEAVER** created by Suresh, Wadwha, Banerjee, and Singh that computes certified probability bounds `[PLB, PUB]` on the likelihood that a <i>local non-autoregressive</i> LLM satisfies a behavioral constraint. Unlike BEAVER that uses vLLM to serve large commercial models, LocalBEAVER allows the user to run their own trained transformer models with their own architectures on verification tasks. 
- BEAVER's **FrontierVerifier** uses branch-and-bound search over the token tree to produce provably correct intervals that tighten with more compute.
- A logits based verifier uses the log probabilities to produce probability bounds as a form of validation.
- Evaluate any binary constraint: security vulnerabilities, toxicity, privacy leakage, hallucination, stereotype, and more.

## 📚 Features

|                                                                                                          |
| -------------------------------------------------------------------------------------------------------- |
| 🔒 Certified `[PLB, PUB]` bounds — the true probability is guaranteed inside the interval                |
| 🌲 Branch-and-bound **FrontierVerifier** reaches tight bounds with far fewer model queries than sampling |
| 📐 Optional EBNF grammar masking (Python, Rust, C, Go) for syntax-constrained generation                 |
| ⚡ SQLite constraint-result caching — avoid re-running expensive checks                                  |
| 🤖 Use any local non-autoregressive model architecture you want                                          |
| 📋 Use the built in NLTK tokenizer, or code up your own.                                                 |
| 🧤 Contains support for GLoVe embeddings                                                                 |
| 🔌 Plug in any binary constraint — one Python function is all you need                                   |

## ⚙️ Requirements

- Python 3.10+

## 🚀 Quick Start

```bash
pip install -e .
```

## 🖥️ Running LocalBEAVER

BEAVER can be used in two ways: as a **Python API** or as a **CLI tool**.

## 👀 Example Usage - CLI

### 1 - Setting Up Your Experiment
First, choose an experiment you would like to run. 
We currently support the following experiments:
```
- conll - the CoNLL 2003 NER task
- reverse - Reversing a list of integers
- sort - Sorting a list of integers
```
and are working on porting over the rest of the experiments from the original BEAVER paper to this version.
You may also additionally write a custom experiment via a custom constraint. See the bottom for more details.
Ensure that your experiment has a valid .yaml config as well, which can also be found at the bottom. 

### 2 - Your Model
LocalBeaver supports to use of both a PyTorch checkpoint as a .pt file and the use of a .py file in the special case of a TransformerProgram. 
When using your model, you must define its architecture and link it to the specific verifier you are using. 
We recommend creating a python file in the beaver/utils folder that contains this architecture. 
The forward() command is required for all architectures.
Once you have created this file, find the Python file for the verifier you will be using. 
For Frontier, this is beaver/verifiers/frontier_verifier.py
When declaring the class, you will see that it loads the model here. 
Use the Python file with your architecture to load your PyTorch checkpoint and pass it forward through the program.

Please note, LocalBeaver currently only supports non-autoregressive models, or models that produce all logits at once and independently of other logits.
We are working to bring support for autoregressive models as well.

### 3 - Your Model Arguments
As you will be implementing your own architecture, you may need arguments about <i>how</i> the model was learned to avoid dimension conflicts. 
For the sort, reverse, and conll tasks, this is the case. 
We provide the model_args parameter to solve this issue.
It expects a JSON object that contains key value pairs with information about the dimensionality of layers of the model.
You may find it necessary to use this for your architecture as well.
For any model generated by the TransformerProgram (TPM) code, you will need the corresponding .json file generated after learning the model.
That is the input for the model_args parameter.

### 4 - Running Verification
LocalBeaver features an increased number of parameters due to the nature of local models. 
To run beaver, see the following command as an example:
```
beaver run
    --experiment experiments/sort/sort.py
    --model experiments/sort/sort.pt
    --model_type transformer
    --model_args experiments/sort/learning_args.json
```
All parameters shown here are <b>required</b>. 
There are many other parameters, and a full list can be found in beaver/cli.py.

Batch commands can also be used to run experiments. 
A YAML file is required for the config of this batch experiment. 
An example can be found at configs/batches/tpm_batch.yaml.
Use the following example command to run a batch experiment:
```
beaver rct-batch --batch configs/batches/tpm_batch.yaml
```

## 👀 Example Usage - Python API
LocalBeaver can also be ran as an API. 
When ran as an API, it requires all required parameters the CLI requires, plus those required parameters that the experimental .py file provides.
This includes the parameters:
```
prompts - a dict, contains your inputs, with a minimum of a "inputs" key with your input for each
constraint_fn - the function used to check the constraint as binary
```

For example:
```python
import beaver

results = beaver.run(
    prompts=[{"inputs": ["2", "3", "1"]}],
    constraint_fn=lambda instance, seq: valid_prefix(seq),  # Your prefix closed constraint here
    experiment=experiments/sort/sort.py,
    model=experiments/sort/sort.pt,
    model_type=transformer,
    model_args=experiments/sort/learning_args.json
)

# Each result: {"idx": 0, "lower_bound": 0.91, "upper_bound": 0.97, "transition": 42, ...}
```

## 📋 Logging & Results

Every BEAVER run creates a timestamped log directory (default: `logging/logs_<timestamp>/`) containing detailed outputs for inspection and post-hoc analysis.

### Log directory structure

```
logging/logs_20260316185322/
├── run_args.json            # Full configuration used for this run
├── bounds.csv               # Per-instance lower/upper bounds and transition counts
├── summary.json             # Aggregate statistics across all instances
├── console.log              # Full console output captured during the run
├── profiling_summary.json   # Timing breakdowns (per-transition and per-instance)
├── <idx>.jsonl              # Per-instance transition log (one JSON object per step)
└── <idx>.profile.json       # Per-instance timing profile (one entry per step)
```

### Key files

- **`bounds.csv`** — Quick overview of results. Each row contains `idx`, `lower_bound`, `upper_bound`, and `num_transitions` for one instance.
- **`summary.json`** — Aggregated metrics: average bounds, transition counts, and constraint satisfaction rates at a configurable threshold.
- **`<idx>.jsonl`** — Detailed per-transition log for instance `idx`, including expanded tokens, decoded text, current bounds, and frontier state at each step.
- **`<idx>.profile.json`** — Timing breakdown per transition (model generation, grammar masking, semantic checks, frontier updates, etc.).

### Summarizing logs after a run

Use the `beaver logs` CLI command to re-summarize any previous run:

```bash
beaver logs logging/logs_20260316185322/
```

This prints aggregate statistics (average bounds, transition counts, constraint satisfaction) and saves `summary.json` and `profiling_summary.json` to the log directory.

## 📦 Built-in Experiments

LocalBEAVER supports the following experiments:
```
- conll - the CoNLL 2003 NER task
- reverse - Reversing a list of integers
- sort - Sorting a list of integers
```
All necessary files, including models and example run scripts, are given for these experiments.
All experiments from the original Beaver code are not supported by LocalBeaver, however, we are working to add support for them. 

## ✍️ Writing a Custom Constraint

Each prompt dict must have a `"inputs"` key (string). Prompts here are not required as for LocalBeaver, transformers are expected to be explicitly trained on a task, contrary to LLMs that take in typically a sequence of words (tokens) and output a sequence of words. 

An example of the reverse task Python experiment file would be:
```python
# experiments/my_eval/my_eval.py

def load_prompts() -> list[dict]:
    return [
        {
            "inputs": ["2", "3", "1"],
            "tags": ["1", "3", "2"],
        }
    ]

def constraint_fn(instance: dict, sequence: str) -> bool:
    """True = acceptable, False = violation."""
    return sequence.split() == instance["tags"]

def check_call_fn(instance, decoded_sequences, token_lists):
    """Optional fast pre-filter — skip expensive checks on short prefixes."""
    pass

def instance_context_fn(instance: dict) -> str:
    """Cache key for this instance's constraint context."""
    pass
```

```yaml
# experiments/my_eval/experiment.yaml
experiment_file: my_eval.py
load_prompts_fn: load_prompts
constraint_fn: constraint_fn
check_call_fn: check_call_fn
instance_context_fn: instance_context_fn
cache: true
cache_dataset_name: my_eval
verifier: frontier
gen_length: 32
epsilon: 0.05
max_workers: 8
```

<details>
<summary><b>LocalBEAVER Non-Required Arguments</b></summary>

| Argument                          | Default           | Description                                                           |
| --------------------------------- | ----------------- | --------------------------------------------------------------------- |
| `verifier`                        | `frontier`        | `frontier` (branch-and-bound) or `logits` (raw log probabilities)     |
| `gen_length`                      | `32`              | Max tokens per generated sequence                                     |
| `epsilon`                         | `0.01`            | Convergence threshold — stop when `PUB − PLB ≤ ε`                     |
| `max_iterations`                  | `100`             | Hard iteration cap per instance                                       |
| `max_frontier_size`               | `10000`           | Max partial sequences tracked in the frontier                         |
| `max_frontier_prob`               | `1.0`             | Max cumulative probability tracked in the frontier                    |
| `frontier_scoring_strategy`       | `highest-prob`    | `highest-prob` \| `length-bias` \| `random-select` \| `sample-select` |
| `num_logprobs`                    | `100`             | Top-k log-probs requested per expansion step                          |
| `use_grammar`                     | `false`           | Apply EBNF grammar mask during expansion                              |
| `grammar`                         | —                 | `rust` \| `python` \| `c` \| `go` \| path to `.lark` file             |
| `temperature` / `top_p` / `top_k` | `1.0 / 0.99 / -1` | Sampling distribution parameters                                      |
| `fewshot_messages`                | `[]`              | Global few-shot examples (can be overridden per-instance via `fewshot_messages` key) |
| `max_workers`                     | `16`              | Parallel worker processes                                             |
| `cache`                           | `false`           | Enable SQLite constraint-result cache                                 |
| `start_idx`                       | `0`               | The start index of the dataset to run                                 |
| `end_idx`                         | `len(dataset)`    | The end index of the dataset to run (default is entire dataset)       |
| `debug_ids`                       | `[]`              | Specific ids of the dataset to run                                    |
| `verbose`                         | `false`           | Prints additional debugging information                               |
| `glove_embed`                     | `false`           | Enable the use of GLoVe embeddings for inputs                         |
| `gpu_uuid`                        | `None`            | Enable use of specific GPU - default is use of CPU instead            |
| `seed`                            | `0`               | Seed used for generation of random values                             |
| `vocab_size`                      | `None`            | Enables use of top-(vocab_size) most frequent tokens, when not used, None signals to include all tokens that occur at least once |

</details>

## 😢 Known Issues
This project is still in the developmental phase and therefore is missing many features that generalize this algorithm.
We ask to direct questions or help to David Broczkowski, reachable at broczkod@lafayette.edu. 
The following issues persist in the project:
- Lack of support for experiments provided in the original Beaver publication.
- Model architecture should have an easier and more streamlined way to be used - the user should not have to create python files and manually add their architecture in.
- LocalBeaver should dynamically decide which arguments it needs from model_args when calling the architecture
- vLLM support should be readded so that users do not have to use multiple versions of Beaver.
- The "Sampling" verifier is currently non-functional.
- Support should be expanded to autoregressive models and models that utilize prompts instead of an explicitly trained task. 

We apologize for these shortcomings and hope to provide these in the future. 

## ⚠️ Disclaimer
LocalBeaver makes several modifications to Beaver covered under CC License BY-SA 4.0. This includes:
- the removal of vLLM
- the addition of various model architectures
- the addition of embeddings, encoders, and decoders
- the modification of logit fetching for non-autoregressive models
- the modification of the CLI and API
- the modification of the README.md
- the addition of 3 additional public experiments

Beaver's code and README.md was expanded on in LocalBeaver's project development and, therefore, some code or text may be identical.
For the README, some text was kept as is to preserve the intent of the authors, especially as this project is NOT a standalone project from Beaver and is instead a modification of it.

Additionally, this project holds support for [Transformer Programs, a technology developed by Friedman, Wettig, and Chen](https://github.com/princeton-nlp/TransformerPrograms) and used code that was written by [Pavel Greshnikov and Patrick Nossol which introduced Z3 verification to Transformer Programs](https://github.com/verified-ai/TransformerProgramVerification). 
In this project, Transformer Program Models may be referenced as Transformer Programs, or as TPM. 

This project is support by the [UA Ruhr Research Center Trustworthy Data Science and Security](https://rc-trust.ai/) and by [UA Ruhr](https://www.uaruhr.de/en/) as the work for this project was supported by the UA Ruhr Fellowship. I give my sincerest thanks to these organizations for their support and guidance of this project.

## 🔖 Citation

```bibtex
@misc{suresh2025beaverefficientdeterministicllm,
      title={BEAVER: An Efficient Deterministic LLM Verifier},
      author={Tarun Suresh and Nalin Wadhwa and Debangshu Banerjee and Gagandeep Singh},
      year={2025},
      eprint={2512.05439},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.05439},
}
```

<a href="https://github.com/uiuc-focal-lab/Beaver">Beaver</a> © 2026 by <a href="https://nalinwadhwa02.github.io">Nalin Wadhwa</a> is licensed under <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>
