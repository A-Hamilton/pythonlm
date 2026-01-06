# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**PythonCoder** is a 1.08B parameter Python-only code generation LLM built with JAX/Flax NNX.
**Optimized for Google Colab TPU v6e-1 (single chip, 32GB HBM).**

**Architecture**: Diff-Llama-MTP (Differential Attention + GQA + Multi-Token Prediction)
**Framework**: JAX 0.8.x + Flax NNX 0.12+ + Optax
**GitHub**: https://github.com/A-Hamilton/pythonlm

## Architecture v2 (2026 SOTA - Benchmark Justified)

### What We Use (High ROI)

| Feature | Benefit | Source |
|---------|---------|--------|
| **Differential Attention** | 35-40% fewer params for same quality | [Microsoft DIFF (ICLR 2025)](https://arxiv.org/abs/2410.05258) |
| **MTP (1 token)** | +12-17% on HumanEval/MBPP, enables speculative decoding | [Meta MTP](https://arxiv.org/abs/2404.19737) |
| **YaRN** | Free 4x context extension (train 2K, infer 8K) | [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) |
| **GQA 3:1** | Memory-efficient KV cache | LLaMA 2/3 |
| **WSD Scheduler** | Flexible multi-epoch training | [MiniCPM](https://arxiv.org/abs/2404.06395) |
| **Dropout (0.1)** | Multi-epoch degradation protection | [NeurIPS 2023](https://arxiv.org/abs/2305.13230) |

### What We Removed (Low ROI at 1B)

| Feature | Reason | Source |
|---------|--------|--------|
| **MoE** | Not worthwhile at <1B params, needs 5T+ tokens | [HuggingFace MoE](https://huggingface.co/blog/moe) |
| **MLA** | Designed for 671B scale, overkill for 1B | DeepSeek-V3 |
| **HLP** | Requires FIM data pipeline (not implemented) | Disabled |
| **MTP=4** | OOM at 1B scale, MTP=1 optimal | Experiments |

### Model Configuration (1.08B Parameters)

```python
# TPU v6e Optimized (all dims 256-aligned)
hidden_size = 1536           # 1536 / 256 = 6 MXU tiles
num_layers = 24
num_attention_heads = 12     # head_dim = 128 (LLaMA 3 standard)
num_kv_heads = 4             # GQA 3:1 ratio
intermediate_size = 4096     # 4096 / 256 = 16 tiles
vocab_size = 151808          # Qwen2.5-Coder + 256 alignment
max_position_embeddings = 8192
dropout_prob = 0.1           # Multi-epoch protection
```

## Quick Start (Colab TPU v6e-1)

```bash
# Train on TPU v6e-1
python train_v2.py

# Auto-configured for:
# - Model: 1.08B (Diff-Llama-MTP)
# - Hardware: TPU v6e-1 (32GB HBM)
# - Context: 2048 tokens (training), 8192 (inference via YaRN)
# - Schedule: WSD (90% stable, 10% decay)
# - Epochs: 4 data epochs with dropout protection
```

## File Structure

```
python only model/
├── model_v2.py           # Model architecture (~1010 lines)
├── train_v2.py           # Training script (~600 lines)
├── inference_v2.py       # Generation with KV cache
├── train_colab.ipynb     # Colab notebook
├── qwen_tokenizer/       # Tokenizer files
├── preprocessed_data/    # Training data (~44GB, 2390 shards, ~24B tokens)
└── CLAUDE.md             # This file
```

## Training Configuration

### WSD Scheduler (NOT Cosine Decay)

```python
# Warmup-Stable-Decay allows flexible multi-epoch training
# Source: MiniCPM (arXiv:2404.06395)

# Phase 1: Warmup (linear ramp)
warmup_steps = 2000

# Phase 2: Stable (constant LR - key innovation)
stable_ratio = 0.9  # 90% of training

# Phase 3: Decay (cosine cooldown)
decay_ratio = 0.1   # Final 10%
```

### Batch Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| micro_batch_size | 8 | Memory-constrained for 1.08B |
| gradient_accumulation | 8 | Proper accumulation (see bug fix) |
| effective_batch_size | 64 | 8 × 8 = 64 |
| max_seq_len | 2048 | Train short, extend via YaRN |
| steps_per_epoch | 5000 | Checkpoint interval |

### Multi-Epoch Strategy

```python
# Source: "To Repeat or Not To Repeat" (NeurIPS 2023)
# https://arxiv.org/abs/2305.13230

# Key findings:
# - 4 epochs generally safe before degradation
# - Dropout is the ONLY effective regularization
# - Validation loss unreliable - use HumanEval on checkpoints

target_epochs = 4
dropout_prob = 0.1  # Enabled in model
```

## Critical Bug Fixes (January 2026)

### Bug 1: Dropout Never Activated

**Problem**: Model's `__call__` had `deterministic=True` default, so dropout never fired during training.

**Fix**: Pass `deterministic=False` in training:
```python
# WRONG (old code)
output = model(batch['input_ids'], labels=batch['labels'])

# CORRECT (fixed)
output = model(batch['input_ids'], labels=batch['labels'], deterministic=False)
```

**Source**: [Flax NNX Docs](https://flax.readthedocs.io/en/latest/nnx_basics.html)

### Bug 2: Gradient Accumulation Broken

**Problem**: `optimizer.update()` was called inside the micro-batch loop, updating 8x per step instead of once.

**Fix**: Accumulate gradients, then update once:
```python
# WRONG (old code - updates 8x per step!)
for _ in range(gradient_accumulation_steps):
    loss = train_step(model, optimizer, batch)  # Calls optimizer.update()!

# CORRECT (fixed - accumulate then update once)
accumulated_grads = None
for _ in range(gradient_accumulation_steps):
    loss, grads = compute_grads(model, batch)  # No update
    accumulated_grads = accumulate(accumulated_grads, grads)

averaged_grads = scale(accumulated_grads, 1/gradient_accumulation_steps)
apply_grads(model, optimizer, averaged_grads)  # Single update
```

**Source**: [Optax Gradient Accumulation](https://optax.readthedocs.io/en/latest/_collections/examples/gradient_accumulation.html)

## Critical API Patterns (Flax NNX)

```python
# 1. Optimizer MUST have wrt=nnx.Param
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

# 2. optimizer.update() MUST include model
optimizer.update(model, grads)

# 3. Parameter access (avoid .value deprecation)
param_value = param[...]  # CORRECT
param_value = param.value  # DEPRECATED

# 4. Dropout requires deterministic=False for training
output = model(inputs, deterministic=False)  # Training
output = model(inputs, deterministic=True)   # Inference
```

## Differential Attention (Numerical Stability)

```python
# Layer-depth-dependent lambda (DIFF Transformer paper)
lambda_init = 0.8 - 0.6 * exp(-0.3 * layer_idx)

# Small init to prevent exp overflow
lambda_q1 = nnx.Param(normal(head_dim) * 0.01)  # NOT 0.1!

# Clamp dot products and final lambda
dot1 = jnp.clip(dot1, -10.0, 10.0)
lambda_val = jnp.clip(lambda_val, 0.0, 1.0)

# Floor attention at 0 (can go negative from subtraction)
diff_attn = jnp.maximum(attn1 - lambda_val * attn2, 0.0)
```

## TPU v6e Optimization

| Setting | Value | Reason |
|---------|-------|--------|
| Dimension alignment | 256 | v6e MXU is 256x256 |
| vocab_size | 151,808 | 256-aligned |
| intermediate_size | 4,096 | 256-aligned (16 tiles) |
| hidden_size | 1,536 | 256-aligned (6 tiles) |
| bfloat16 | Yes | Native TPU dtype |

## Checkpointing (Orbax)

```python
import orbax.checkpoint as ocp
from flax import nnx

# Save with metadata
state = {
    'model': nnx.state(model),
    'optimizer': nnx.state(optimizer),
    'epoch': epoch,
    'global_step': global_step,
    'data_epoch': data_epoch,  # Track which pass through data
}
checkpointer = ocp.StandardCheckpointer()
checkpointer.save(path, state)
checkpointer.wait_until_finished()

# Restore
abstract_state = jax.tree.map(ocp.utils.to_shape_dtype_struct, state)
restored = checkpointer.restore(path, abstract_state)
nnx.update(model, restored['model'])
```

## Scaling Math

For 1.08B parameters with ~24B tokens:
- **Chinchilla ratio**: 24B / 1.08B = 22x (optimal is 20x) ✓
- **Tokens per step**: 2048 × 64 = 131,072
- **1 data epoch**: 24B / 131,072 = 183,105 steps
- **4 data epochs**: 732,420 steps total
- **Tokens/param (4 epochs)**: 88x

## Benchmarks (Realistic Targets)

| Benchmark | Base Model Target | With Inference Scaling |
|-----------|-------------------|------------------------|
| HumanEval | 45-50% | 70-75% (best-of-10) |
| MBPP | 40-45% | 60-65% |

**Note**: Qwen2.5-Coder-1.5B achieves 46.8% HumanEval base. Our 1.08B target is comparable.

## Known Issues

### 1. NaN Loss
**Cause**: Differential attention lambda overflow
**Fix**: Clamp dot products to [-10, 10], lambda to [0, 1], init scale 0.01

### 2. Hugepages Warning (TPU)
**Message**: "Transparent hugepages are not enabled"
**Impact**: Non-critical, only affects startup time

### 3. Context7 Quota
**Issue**: MCP tool quota exceeded
**Workaround**: Use WebSearch for documentation lookups

## References

- [Differential Transformer (ICLR 2025)](https://arxiv.org/abs/2410.05258)
- [Multi-Token Prediction (Meta)](https://arxiv.org/abs/2404.19737)
- [MiniCPM WSD Schedule](https://arxiv.org/abs/2404.06395)
- [To Repeat or Not To Repeat (NeurIPS 2023)](https://arxiv.org/abs/2305.13230)
- [Flax NNX Documentation](https://flax.readthedocs.io/en/latest/nnx_basics.html)
- [Optax Gradient Accumulation](https://optax.readthedocs.io/en/latest/_collections/examples/gradient_accumulation.html)
- [Qwen2.5-Coder](https://arxiv.org/abs/2409.12186)
