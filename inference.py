"""PythonCoder Inference v2 - Minimal Build (January 2026)

Inference script for Diff-Llama-MTP architecture.
Uses model's built-in generate() with KV cache.
"""

import os
from pathlib import Path

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
        'optimizer': jax.tree.map(ocp.utils.to_shape_dtype_struct, model_state),
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

def generate_code(
    model: PythonCoderModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> str:
    """Generate code from a prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors="jax")

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
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
) -> str:
    """Generate code with Fill-in-Middle."""
    fim_prompt = (
        f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    )
    return generate_code(
        model, tokenizer, fim_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


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
