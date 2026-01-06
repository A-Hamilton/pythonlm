"""HumanEval evaluation for PythonCoder checkpoints.

Evaluates each epoch checkpoint with fixed prompts/decoding settings and logs
pass@1 (and optional pass@k) for model selection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from transformers import AutoTokenizer

from model import CONFIG_1B, PythonCoderModel, create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HumanEval over checkpoints")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing epoch_* checkpoints.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        required=True,
        help="Tokenizer directory (e.g., qwen_tokenizer).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate per sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (fixed across checkpoints).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling value.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for generation.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Samples per HumanEval problem (for pass@k).",
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        nargs="*",
        default=[1],
        help="Compute pass@k for these k values.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("humaneval_results.jsonl"),
        help="Where to append per-checkpoint results.",
    )
    return parser.parse_args()


def list_epoch_checkpoints(checkpoint_dir: Path) -> list[Path]:
    checkpoints = sorted(
        (path for path in checkpoint_dir.glob("epoch_*") if path.is_dir()),
        key=lambda path: int(path.name.split("_")[1]),
    )
    return checkpoints


def load_checkpoint_epoch(model: PythonCoderModel, checkpoint_path: Path) -> int:
    model_state = nnx.state(model)
    abstract_state = {
        "model": jax.tree.map(ocp.utils.to_shape_dtype_struct, model_state),
        "epoch": 0,
        "loss": 0.0,
    }

    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(checkpoint_path, abstract_state)
    nnx.update(model, restored["model"])
    return int(restored.get("epoch", checkpoint_path.name.split("_")[1]))


def ensure_humaneval_available() -> None:
    try:
        import human_eval  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: human_eval. Install with `pip install human-eval`."
        ) from exc


def build_model(tokenizer_dir: Path) -> tuple[PythonCoderModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True
    )
    rngs = nnx.Rngs(42)
    model = create_model(CONFIG_1B, rngs=rngs)
    return model, tokenizer


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples == 0:
        return 0.0
    if k > num_samples:
        return float("nan")
    if num_correct == 0:
        return 0.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(
        num_samples, k
    )


def generate_completion(
    model: PythonCoderModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    rng: jax.Array,
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
        completion = decoded[len(prompt):]
    else:
        completion = decoded
    return completion.split("\n\n", 1)[0]


def evaluate_checkpoint(
    model: PythonCoderModel,
    tokenizer: AutoTokenizer,
    num_samples: int,
    pass_k: Iterable[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> dict[int, float]:
    from human_eval.data import read_problems
    from human_eval.execution import check_correctness

    problems = read_problems()
    rng = jax.random.PRNGKey(seed)

    per_problem_correct: list[int] = []

    for task_id, problem in problems.items():
        correct = 0
        for sample_index in range(num_samples):
            rng, step_rng = jax.random.split(rng)
            completion = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=problem["prompt"],
                rng=step_rng,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            result = check_correctness(problem, completion, timeout=3.0)
            if result["passed"]:
                correct += 1
        per_problem_correct.append(correct)

    results = {}
    for k in pass_k:
        scores = [
            estimate_pass_at_k(num_samples, correct, k)
            for correct in per_problem_correct
        ]
        results[k] = float(jnp.nanmean(jnp.array(scores)))
    return results


def main() -> None:
    args = parse_args()
    ensure_humaneval_available()

    checkpoints = list_epoch_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        raise SystemExit(f"No epoch_* checkpoints in {args.checkpoint_dir}")

    if max(args.pass_k, default=1) > args.num_samples:
        print(
            "WARNING: pass@k requested with k > num_samples; results will be NaN."
        )

    model, tokenizer = build_model(args.tokenizer_dir)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HumanEval evaluation")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print(f"Decoding: temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    print(f"Samples per task: {args.num_samples}")
    print(f"Pass@k: {args.pass_k}")
    print("=" * 80)

    with args.output_jsonl.open("a", encoding="utf-8") as handle:
        for checkpoint in checkpoints:
            epoch = load_checkpoint_epoch(model, checkpoint)
            print(f"Evaluating epoch {epoch} ({checkpoint})...")
            scores = evaluate_checkpoint(
                model=model,
                tokenizer=tokenizer,
                num_samples=args.num_samples,
                pass_k=args.pass_k,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
            )
            record = {
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "num_samples": args.num_samples,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "seed": args.seed,
                "pass_at": scores,
            }
            handle.write(json.dumps(record) + "\n")
            print(f"Epoch {epoch} scores: {scores}")


if __name__ == "__main__":
    main()
