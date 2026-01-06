"""Run HumanEval on a specified checkpoint and record pass@k results.

Example:
    python eval/run_humaneval.py --checkpoint epoch_4 \
        --output eval/humaneval_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import jax
import orbax.checkpoint as ocp
from flax import nnx
from transformers import AutoTokenizer

from model import CONFIG_1B, create_model

try:
    from human_eval.data import read_problems, write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "Missing dependency: human_eval. Install with `pip install human-eval`."
    ) from exc


PROMPT_TEMPLATE = "{prompt}"

DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HumanEval and log pass@k.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint name (e.g. epoch_4 or 4) or path to checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints_v2",
        help="Directory containing checkpoint folders (default: checkpoints_v2).",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default="qwen_tokenizer",
        help="Tokenizer directory (default: qwen_tokenizer).",
    )
    parser.add_argument(
        "--output",
        default="eval/humaneval_results.json",
        help="Output JSON/CSV file for results.",
    )
    parser.add_argument(
        "--pass-k",
        default="1",
        help="Comma-separated k values for pass@k (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for decoding.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Max new tokens (default: {DEFAULT_MAX_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Top-p nucleus sampling (default: {DEFAULT_TOP_P}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Top-k sampling (default: {DEFAULT_TOP_K}).",
    )
    return parser.parse_args()


def resolve_checkpoint(checkpoint: str, checkpoint_root: str) -> Path:
    candidate = Path(checkpoint)
    if candidate.exists():
        return candidate

    root = Path(checkpoint_root)
    if checkpoint.isdigit():
        candidate = root / f"epoch_{checkpoint}"
    else:
        candidate = root / checkpoint

    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    return candidate


def load_checkpoint(model, checkpoint_path: Path) -> dict:
    model_state = nnx.state(model)
    abstract_state = {
        "model": jax.tree.map(ocp.utils.to_shape_dtype_struct, model_state),
    }
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(checkpoint_path, abstract_state)
    nnx.update(model, restored["model"])
    return restored


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    rng,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="jax")
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        rng=rng,
    )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        return decoded[len(prompt):]
    return decoded


def parse_pass_k(value: str) -> list[int]:
    ks = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        ks.append(int(item))
    if not ks:
        raise ValueError("pass-k must include at least one value")
    return sorted(set(ks))


def write_results_json(path: Path, checkpoint_key: str, payload: dict) -> None:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = {}
    data[checkpoint_key] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def write_results_csv(path: Path, checkpoint_key: str, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["checkpoint"] + sorted(
        [key for key in payload if key.startswith("pass@")]
    )
    row = {"checkpoint": checkpoint_key, **payload}
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def record_results(output_path: Path, checkpoint_key: str, payload: dict) -> None:
    if output_path.suffix.lower() == ".csv":
        write_results_csv(output_path, checkpoint_key, payload)
    else:
        write_results_json(output_path, checkpoint_key, payload)


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint, args.checkpoint_root)
    checkpoint_key = checkpoint_path.name
    ks = parse_pass_k(args.pass_k)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    rngs = nnx.Rngs(args.seed)
    model = create_model(CONFIG_1B, rngs=rngs)
    load_checkpoint(model, checkpoint_path)

    problems = read_problems()
    samples_path = (
        Path("eval")
        / f"humaneval_samples_{checkpoint_key}_{int(time.time())}.jsonl"
    )

    samples = []
    rng = jax.random.PRNGKey(args.seed)
    for task_id, problem in problems.items():
        rng, step_rng = jax.random.split(rng)
        prompt = PROMPT_TEMPLATE.format(prompt=problem["prompt"])
        completion = generate_completion(
            model,
            tokenizer,
            prompt,
            step_rng,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        samples.append({"task_id": task_id, "completion": completion})

    samples_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(samples_path, samples)

    results = evaluate_functional_correctness(
        samples_path,
        k=ks,
        n_workers=4,
        timeout=3.0,
    )

    payload = {
        "checkpoint": checkpoint_key,
        "prompt_template": PROMPT_TEMPLATE,
        "decoding": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "num_tasks": len(problems),
        "samples_file": str(samples_path),
        **results,
    }

    output_path = Path(args.output)
    record_results(output_path, checkpoint_key, payload)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
