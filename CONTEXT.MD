# Project Context: Orchestrator + Specialist Code Models

## Vision

Build a multi-model AI coding system where:
- **Orchestrator model** (3-7B params): Breaks down tasks, routes to specialists, combines results
- **Specialist models** (500M-1B params each): Language-specific experts trained only on that language

```
User Query: "Build a web scraper in Python that saves to Postgres"
                              │
                              ▼
                    ┌─────────────────┐
                    │   Orchestrator  │
                    │    (3-7B)       │
                    │                 │
                    │ "Break this     │
                    │  into tasks,    │
                    │  route to       │
                    │  specialists"   │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │   Python    │  │   Python    │  │    SQL      │
    │  Specialist │  │  Specialist │  │  Specialist │
    │   (500M)    │  │   (500M)    │  │   (500M)    │
    │             │  │             │  │             │
    │ "Web        │  │ "DB         │  │ "Design     │
    │  scraper    │  │  connection │  │  schema"    │
    │  code"      │  │  code"      │  │             │
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Orchestrator  │
                    │   (combines)    │
                    └────────┬────────┘
                             │
                             ▼
                      Final Response
```

## Why This Architecture?

### Cost Comparison

| Approach | Inference Cost | Training Cost |
|----------|---------------|---------------|
| Single 70B model (GPT-4 style) | $0.01-0.03/query | $500,000+ |
| Our system (orchestrator + specialists) | $0.0001-0.001/query | $5,000-10,000 |

**250-750x cheaper inference, 50-100x cheaper training**

### Technical Advantages

1. **True specialization**: Each model trained ONLY on its language (not just prompted)
2. **Parallel execution**: Run 5 specialists simultaneously
3. **Modular updates**: Retrain just Python model without touching others
4. **Memory efficient**: Load only needed specialists
5. **Cheaper to run**: 500M model vs 70B model per task

## Current Phase: Python Specialist (First Model)

### Why Python First?
- Largest training data available
- Easiest to evaluate (HumanEval, MBPP benchmarks)
- Most immediately useful
- Validates the entire pipeline

### Training Stack

```
Framework:     JAX + Flax
Hardware:      TPU v5e-8 (free via TRC/Kaggle)
Data:          The Stack v2 (Python subset)
Tokenizer:     StarCoder2 (49,152 tokens)
Training:      ~25B tokens for 500M model
Cost:          $0-50 (FREE with TRC)
```

## Model Specifications

### Test Model (10M params) - FOR VALIDATION

```python
TestConfig:
    vocab_size: 49152      # StarCoder2 tokenizer
    hidden_dim: 256
    num_layers: 6
    num_heads: 4
    head_dim: 64           # hidden_dim // num_heads
    ffn_dim: 1024          # 4x hidden_dim
    max_seq_len: 512
    dropout: 0.0
    dtype: bfloat16

Memory: ~200 MB
Training: 10-30 minutes
Purpose: Validate pipeline works end-to-end
```

### Production Model (500M params) - REAL TRAINING

```python
ProdConfig:
    vocab_size: 49152      # StarCoder2 tokenizer
    hidden_dim: 1024
    num_layers: 24
    num_heads: 16
    head_dim: 64           # hidden_dim // num_heads
    ffn_dim: 4096          # 4x hidden_dim
    max_seq_len: 2048
    dropout: 0.0
    dtype: bfloat16

Memory: ~8 GB
Training: 24-48 hours on TPU v5e-8
Purpose: Actual Python specialist
```

## Architecture Details

### Model Type
- **Decoder-only transformer** (GPT-style, causal language model)
- **Next token prediction** objective

### Key Components

| Component | Implementation | Why |
|-----------|---------------|-----|
| Attention | Multi-Query Attention (MQA) | Faster inference, less memory |
| Position | RoPE (Rotary Position Embeddings) | Better length generalization |
| Normalization | RMSNorm | Faster than LayerNorm |
| Activation | SwiGLU | Better than ReLU/GELU |
| Precision | bfloat16 | 2x memory savings, TPU-native |

### Architecture Diagram

```
Input IDs [batch, seq_len]
         │
         ▼
┌─────────────────┐
│    Embedding    │  vocab_size → hidden_dim
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         Transformer Block           │  ×num_layers
│  ┌───────────────────────────────┐  │
│  │  RMSNorm → MQA → Residual     │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │  RMSNorm → SwiGLU → Residual  │  │
│  └───────────────────────────────┘  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│    RMSNorm      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Linear  │  hidden_dim → vocab_size (tied with embedding)
└────────┬────────┘
         │
         ▼
Logits [batch, seq_len, vocab_size]
```

## Data Pipeline

### Source
- **The Stack v2** (Python subset): ~100GB deduplicated Python code
- Alternative: **The Stack v1** or **codeparrot-clean** (easier access)

### Quality Filters

```python
# 1. Size filters
- Minimum 100 characters
- Maximum 1,000,000 characters

# 2. Quality heuristics
- Not auto-generated (no "generated by", "do not edit")
- Not data files (no long hex strings)
- Has meaningful code (not just comments)
- Reasonable line lengths (not minified)

# 3. Syntax validation
- Must parse as valid Python (ast.parse)

# 4. Deduplication
- MinHash LSH with Jaccard threshold 0.7
- File-level (not repo-level)
```

### Data Format

```python
# Each training sample:
{
    'input_ids': [token_1, token_2, ..., token_seq_len],  # int32
}

# Loss computed as:
# - input: tokens[:-1]
# - target: tokens[1:]
# - Cross-entropy loss, masked for padding
```

## Training Configuration

### Optimizer
- **AdamW** with weight decay 0.1
- Gradient clipping: 1.0

### Learning Rate Schedule
```
Warmup: Linear 0 → 3e-4 over 2000 steps
Decay:  Cosine 3e-4 → 3e-5 over remaining steps
```

### Batch Size
- Global batch size: 64 (for 500M model)
- Per-device: 8 (across 8 TPU chips)

### Training Duration
- 10M model: 1,000-5,000 steps (validation only)
- 500M model: 50,000-100,000 steps (~25B tokens)

## Hardware: TPU v5e-8

### Specifications
```
Chips:          8
Memory:         16 GB HBM per chip = 128 GB total
Compute:        ~197 TFLOPS per chip = ~1,576 TFLOPS total (bf16)
Interconnect:   ICI between chips
```

### What It Can Train
| Model Size | Fits? | Notes |
|------------|-------|-------|
| 10M | ✅ Easy | Validation model |
| 100M | ✅ Easy | Quick experiments |
| 500M | ✅ Comfortable | Main specialist |
| 1B | ✅ Good | Larger specialist |
| 3B | ⚠️ Tight | Needs optimization |
| 7B | ⚠️ Very tight | Gradient checkpointing required |
| 13B+ | ❌ No | Need larger slice |

### Access Options
1. **Kaggle** (FREE): TPU v5e-8, 30 hours/week
2. **TRC** (FREE): Apply at sites.research.google/trc, 30 days renewable
3. **Google Cloud** (PAID): ~$1.20/hour for v5e-8

## File Structure

```
project/
├── context.md              # This file - project overview
├── train.py                # Main training script
├── model.py                # Model architecture (JAX/Flax)
├── data.py                 # Data loading and processing
├── config.py               # Model and training configs
├── tokenizer/              # Tokenizer (use StarCoder2)
├── checkpoints/            # Saved model checkpoints
└── outputs/                # Training logs, metrics
```

## Success Criteria

### 10M Model (Validation)
- [ ] Pipeline runs without errors
- [ ] Loss decreases during training
- [ ] Can save and load checkpoints
- [ ] Can generate tokens (quality doesn't matter)

### 500M Model (Production)
- [ ] Final loss: 2.0-2.5
- [ ] HumanEval: 15-25%
- [ ] MBPP: 20-30%
- [ ] Generates syntactically valid Python
- [ ] Useful code completions

## Next Steps After Python Specialist

1. **Evaluate** Python model thoroughly
2. **Train** Rust specialist (same process, different data)
3. **Train** JavaScript specialist
4. **Train** SQL specialist
5. **Train** Orchestrator model (routing logic)
6. **Build** inference system (parallel execution)
7. **Integrate** into unified API

## Dependencies

```
# Core
jax[tpu]>=0.4.20
flax>=0.8.0
optax>=0.1.7

# Data
datasets>=2.14.0
transformers>=4.35.0
tokenizers>=0.15.0

# Utilities
tqdm
numpy
orbax-checkpoint

# Optional
wandb  # Logging
```

## References

- MaxText (Google's JAX LLM): https://github.com/AI-Hypercomputer/maxtext
- The Stack v2: https://huggingface.co/datasets/bigcode/the-stack-v2
- StarCoder2: https://huggingface.co/bigcode/starcoder2-3b
- JAX Documentation: https://jax.readthedocs.io/
- TPU Research Cloud: https://sites.research.google/trc/