"""PythonCoder Inference v2 - Minimal Build (January 2026)

Inference script for Diff-Llama-MTP architecture.
Uses model's built-in generate() with KV cache.
"""

import json
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from transformers import AutoTokenizer

from model import PythonCoderModel, CONFIG_1B, SPECIAL_TOKENS, create_model


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = "/content/drive/MyDrive/python-coder-v6e"
CHECKPOINT_DIR = f"{BASE_DIR}/checkpoints_v2"
TOKENIZER_DIR = "./qwen_tokenizer"


# =============================================================================
# Checkpoint Loading
# =============================================================================

def load_checkpoint(model: PythonCoderModel) -> int:
    """Load latest checkpoint. Returns epoch number."""
    ckpt_dir = Path(CHECKPOINT_DIR)
    if not ckpt_dir.exists():
        print(f"No checkpoint directory: {CHECKPOINT_DIR}")
        return 0

    checkpoints = list(ckpt_dir.glob("epoch_*"))
    if not checkpoints:
        print("No checkpoints found.")
        return 0

    latest = max(checkpoints, key=lambda p: int(p.name.split("_")[1]))
    print(f"Loading: {latest}")

    model_state = nnx.state(model)
    abstract_state = {
        'model': jax.tree.map(ocp.utils.to_shape_dtype_struct, model_state),
        'epoch': 0,
        'loss': 0.0,
    }

    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(latest, abstract_state)
    nnx.update(model, restored['model'])

    return restored['epoch']


# =============================================================================
# Generation
# =============================================================================

def build_decoding_config(
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: Optional[int],
    deterministic: bool,
    best_of: Optional[int] = None,
) -> dict:
    """Build a decoding config dictionary for logging."""
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "deterministic": deterministic,
        "best_of": best_of,
    }


def _make_rng(seed: Optional[int]) -> Optional[jax.Array]:
    if seed is None:
        return None
    return jax.random.PRNGKey(seed)


def generate_ids(
    model: PythonCoderModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    seed: Optional[int] = None,
    deterministic: bool = False,
) -> jax.Array:
    """Generate token IDs from a prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors="jax")

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        deterministic=deterministic,
        rng=_make_rng(seed),
    )

    return output_ids


def generate_code(
    model: PythonCoderModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    seed: Optional[int] = None,
    deterministic: bool = False,
) -> str:
    """Generate code from a prompt."""
    output_ids = generate_ids(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        deterministic=deterministic,
    )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated


def generate_fim(
    model: PythonCoderModel,
    tokenizer,
    prefix: str,
    suffix: str,
    max_new_tokens: int = 128,
    temperature: float = 0.5,
    top_p: float = 0.9,
    top_k: int = 50,
    seed: Optional[int] = None,
    deterministic: bool = False,
) -> str:
    """Generate code with Fill-in-Middle."""
    fim_prompt = (
        f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    )
    return generate_code(
        model, tokenizer, fim_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        deterministic=deterministic,
    )


def _sequence_logprob(
    model: PythonCoderModel,
    input_ids: jax.Array,
    output_ids: jax.Array,
) -> jax.Array:
    """Compute log-probability of generated tokens for best-of-N."""
    out = model(output_ids, deterministic=True)
    logits = out["logits"][:, :-1, :]
    target = output_ids[:, 1:]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_log_probs = jnp.take_along_axis(log_probs, target[..., None], axis=-1).squeeze(-1)
    prompt_len = input_ids.shape[1]
    mask = jnp.arange(target.shape[1]) >= (prompt_len - 1)
    masked = token_log_probs * mask
    return masked.sum(axis=-1)


def best_of_n_generate(
    model: PythonCoderModel,
    tokenizer,
    prompt: str,
    *,
    n: int,
    seed: int,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> dict:
    """Generate best-of-N sample using log-probability scoring."""
    input_ids = tokenizer.encode(prompt, return_tensors="jax")
    rng = jax.random.PRNGKey(seed)
    keys = jax.random.split(rng, n)
    samples = []
    scores = []
    for key in keys:
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            deterministic=False,
            rng=key,
        )
        score = _sequence_logprob(model, input_ids, output_ids)
        samples.append(output_ids)
        scores.append(score)
    scores_arr = jnp.concatenate(scores, axis=0)
    best_idx = int(jnp.argmax(scores_arr))
    best_ids = samples[best_idx]
    best_text = tokenizer.decode(best_ids[0], skip_special_tokens=True)
    return {
        "best_text": best_text,
        "best_score": float(scores_arr[best_idx]),
        "all_scores": [float(score[0]) for score in scores],
    }


def sweep_temperature_top_p(
    model: PythonCoderModel,
    tokenizer,
    prompt: str,
    *,
    temperatures: Sequence[float],
    top_ps: Sequence[float],
    seed: Optional[int] = None,
    max_new_tokens: int = 256,
    top_k: int = 50,
    deterministic: bool = False,
) -> dict:
    """Generate a grid sweep over temperature/top-p."""
    results = {}
    rng = _make_rng(seed)
    total = len(temperatures) * len(top_ps)
    if rng is not None and not deterministic:
        keys = jax.random.split(rng, total)
    else:
        keys = [None] * total
    idx = 0
    for temperature in temperatures:
        for top_p in top_ps:
            key = keys[idx]
            idx += 1
            output_ids = model.generate(
                tokenizer.encode(prompt, return_tensors="jax"),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                deterministic=deterministic,
                rng=key,
            )
            generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            results[(temperature, top_p)] = generated
    return results


def write_humaneval_results(
    output_path: Path,
    results: Iterable[dict],
    decoding_config: dict,
) -> None:
    """Write HumanEval results with decoding config attached per record."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            payload = dict(result)
            payload["decoding_config"] = decoding_config
            handle.write(json.dumps(payload) + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("PythonCoder v2 Inference - Diff-Llama-MTP")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)

    # Create model
    print("Creating model...")
    rngs = nnx.Rngs(42)
    model = create_model(CONFIG_1B, rngs=rngs)

    params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model)))
    print(f"Parameters: {params:,} ({params/1e9:.2f}B)")

    # Load checkpoint
    epoch = load_checkpoint(model)
    if epoch > 0:
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        print("WARNING: No checkpoint loaded, using random weights")

    # Test generation
    print("\n" + "-" * 60)
    print("TEST GENERATION")
    print("-" * 60)

    test_prompt = "def fibonacci(n):"
    print(f"\nPrompt: {test_prompt}")
    print("\nGenerated:")

    output = generate_code(
        model, tokenizer, test_prompt,
        max_new_tokens=128,
        temperature=0.7,
    )
    print(output)

    # Test FIM
    print("\n" + "-" * 60)
    print("TEST FIM (Fill-in-Middle)")
    print("-" * 60)

    prefix = "def add(a, b):\n    "
    suffix = "\n    return result"
    print(f"\nPrefix: {prefix!r}")
    print(f"Suffix: {suffix!r}")
    print("\nGenerated middle:")

    fim_output = generate_fim(model, tokenizer, prefix, suffix)
    print(fim_output)


if __name__ == "__main__":
    main()
