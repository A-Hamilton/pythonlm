# Prompt: Build 10M Parameter Python Code Model for Pipeline Validation

## Task

Create a complete, runnable JAX/Flax training pipeline for a 10M parameter Python code model. This is a validation model to test the entire pipeline before scaling to 500M parameters.

**Target environment**: Kaggle TPU v5e-8 or Google Colab TPU

---

## Requirements

### 1. Model Architecture

Create a decoder-only transformer with these EXACT specifications:

```python
class Config:
    vocab_size: int = 49152      # StarCoder2 tokenizer vocab size
    hidden_dim: int = 256        # Embedding dimension
    num_layers: int = 6          # Number of transformer blocks
    num_heads: int = 4           # Attention heads
    head_dim: int = 64           # hidden_dim // num_heads
    ffn_dim: int = 1024          # Feed-forward dimension (4x hidden)
    max_seq_len: int = 512       # Maximum sequence length
    dropout: float = 0.0         # No dropout for small model
    dtype = jnp.bfloat16         # Use bfloat16 for TPU
```

**Parameter count should be approximately 10-12M.**

### 2. Architecture Components

Implement these specific components:

#### a) RMSNorm (not LayerNorm)
```python
# RMSNorm: x / sqrt(mean(x^2) + eps) * weight
# No bias, no mean subtraction
```

#### b) Rotary Position Embeddings (RoPE)
```python
# Apply rotary embeddings to Q and K in attention
# theta = 10000.0
# Do NOT use learned positional embeddings
```

#### c) Multi-Query Attention (MQA)
```python
# Q: num_heads separate heads
# K: SINGLE head, broadcast to all Q heads  
# V: SINGLE head, broadcast to all Q heads
# This saves memory and is faster
```

#### d) SwiGLU Feed-Forward
```python
# gate = silu(x @ W_gate)
# up = x @ W_up
# output = (gate * up) @ W_down
# Three weight matrices, not two
```

#### e) Causal Attention Mask
```python
# Lower triangular mask
# Prevent attending to future tokens
```

#### f) Tied Embeddings
```python
# Output projection shares weights with input embedding
# logits = hidden @ embedding_weights.T
```

### 3. Training Components

#### a) Optimizer: AdamW
```python
learning_rate = 1e-3          # Higher LR for small model
warmup_steps = 100            # Short warmup
total_steps = 5000            # Quick validation
weight_decay = 0.1
grad_clip = 1.0               # Gradient clipping
beta1 = 0.9
beta2 = 0.95
```

#### b) Learning Rate Schedule
```python
# Warmup: linear from 0 to learning_rate over warmup_steps
# Decay: cosine decay to learning_rate * 0.1
```

#### c) Loss Function
```python
# Cross-entropy loss for next token prediction
# Input: tokens[:-1]
# Target: tokens[1:]
# Mask padding tokens (token_id = 0 or pad_token_id)
```

### 4. Data Pipeline

#### a) Data Source
```python
# Use HuggingFace datasets with streaming
# Primary: "bigcode/the-stack" with data_dir="data/python"
# Fallback: "codeparrot/codeparrot-clean"
```

#### b) Tokenizer
```python
# Use: AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
# Set: tokenizer.pad_token = tokenizer.eos_token
```

#### c) Data Processing
```python
# 1. Filter: 100 < len(content) < 50000 characters
# 2. Tokenize with truncation to max_seq_len + 1
# 3. Pad shorter sequences
# 4. Return {'input_ids': tokens}
```

### 5. Required Functions

```python
def validate_pipeline():
    """
    Run quick checks before training:
    1. Verify TPU devices available
    2. Test tokenizer encode/decode
    3. Test data loading
    4. Test model forward pass
    5. Test loss computation
    6. Test one training step
    7. Verify loss is not NaN
    """
    pass

def train(config, num_steps=5000, log_every=100):
    """
    Main training loop:
    1. Initialize model and optimizer
    2. Create data iterator
    3. Training loop with progress bar
    4. Log loss every log_every steps
    5. Save checkpoint at end
    """
    pass

def generate(state, tokenizer, prompt, max_tokens=50, temperature=0.8):
    """
    Generate code completion:
    1. Tokenize prompt
    2. Autoregressive generation
    3. Sample with temperature
    4. Stop at EOS or max_tokens
    5. Decode and return string
    """
    pass
```

### 6. Output Requirements

The script should:

1. **Print device info** at startup:
```
JAX version: x.x.x
Devices: [TpuDevice(...), ...]
Device count: 8
```

2. **Print model info**:
```
Model parameters: 10,xxx,xxx
```

3. **Print training progress**:
```
Step 100, Loss: 8.2345
Step 200, Loss: 6.1234
...
```

4. **Run validation** before training:
```
✓ Tokenizer works
✓ Data loads  
✓ Model initialized: 10,xxx,xxx params
✓ Forward pass works
✓ Loss computes: x.xxxx
✓ Training step works
```

5. **Test generation** after training:
```
Prompt: def hello():
Generated: def hello():
    print("Hello, World!")
```

---

## File Structure

Create a SINGLE Python file `train_10m.py` that can be run directly in a Kaggle/Colab notebook.

Structure:
```python
#!/usr/bin/env python3
"""
10M Parameter Python Code Model - Pipeline Validation
Run on Kaggle TPU v5e-8 or Google Colab TPU
"""

# ============ IMPORTS ============

# ============ CONFIGURATION ============

# ============ MODEL COMPONENTS ============
# - RMSNorm
# - RotaryEmbedding  
# - Attention (MQA)
# - FeedForward (SwiGLU)
# - TransformerBlock
# - PythonCodeModel

# ============ TRAINING UTILITIES ============
# - create_train_state
# - compute_loss
# - train_step (jitted)

# ============ DATA LOADING ============
# - load_data

# ============ GENERATION ============
# - generate

# ============ VALIDATION ============
# - validate_pipeline

# ============ MAIN TRAINING ============
# - train

# ============ ENTRY POINT ============
if __name__ == "__main__":
    validate_pipeline()
    state = train()
    # Test generation
```

---

## Constraints

1. **Single file**: Everything in one `.py` file for easy notebook use
2. **No external dependencies** beyond standard ML stack (jax, flax, optax, transformers, datasets)
3. **Must work on TPU**: Use bfloat16, avoid operations that don't work on TPU
4. **Clear error messages**: If something fails, print helpful debug info
5. **Reproducible**: Set random seeds
6. **Memory efficient**: Use jax.jit, don't create unnecessary copies

---

## Testing Checklist

After creating the code, verify:

- [ ] Imports work without errors
- [ ] Config creates ~10M params (verify with param count)
- [ ] Model forward pass produces correct output shape: (batch, seq, vocab)
- [ ] Loss is a scalar, not NaN
- [ ] Loss decreases over 100 steps
- [ ] Generation produces valid token sequences
- [ ] Checkpointing saves without error
- [ ] Full training completes in <30 minutes on TPU v5e-8

---

## Example Usage

```python
# In Kaggle notebook:

# Cell 1: Install
!pip install -q jax[tpu] flax optax transformers datasets tqdm

# Cell 2: Paste the entire train_10m.py content

# Cell 3: Run
validate_pipeline()  # Should show all ✓
state = train(num_steps=1000)  # Quick test

# Cell 4: Generate
print(generate(state, tokenizer, "def fibonacci(n):"))
```

---

## Notes for the Agent

1. **Start simple**: Get forward pass working first, then add training
2. **Test incrementally**: Don't write everything then debug
3. **Use existing implementations as reference**: StarCoder2, LLaMA architectures
4. **bfloat16 everywhere**: TPUs are optimized for bf16
5. **MQA is important**: Single K/V head, multiple Q heads - this is not standard multi-head attention

The goal is a WORKING pipeline, not a high-quality model. The 10M model will generate garbage - that's fine. We just need to verify the pipeline works before scaling to 500M.