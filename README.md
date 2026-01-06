# PythonLM

A 1.08B parameter Python-only code generation model built with JAX/Flax NNX.

**Architecture**: Diff-Llama-MTP (Differential Attention + GQA + Multi-Token Prediction)
**Hardware**: Optimized for Google Colab TPU v6e-1 (32GB HBM)

## Architecture Features (2026 SOTA, Benchmark-Justified)

| Feature | Benefit | Source |
|---------|---------|--------|
| **Differential Attention** | 35-40% fewer params for same quality | [Microsoft DIFF (ICLR 2025)](https://arxiv.org/abs/2410.05258) |
| **Multi-Token Prediction** | +12-17% on HumanEval/MBPP | [Meta MTP](https://arxiv.org/abs/2404.19737) |
| **GQA (3:1)** | Memory-efficient KV cache | LLaMA 2/3 |
| **YaRN** | 4x context extension (train 2K, infer 8K) | [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) |
| **WSD Scheduler** | Flexible multi-epoch training | [MiniCPM](https://arxiv.org/abs/2404.06395) |

## Model Configuration

```python
hidden_size = 1536         # 256-aligned for TPU v6e MXU
num_layers = 24
num_attention_heads = 12   # head_dim = 128 (LLaMA 3 standard)
num_kv_heads = 4           # GQA 3:1 ratio
intermediate_size = 4096   # SwiGLU FFN
vocab_size = 151808        # Qwen2.5-Coder tokenizer
max_position_embeddings = 8192
```

**Parameters**: ~1.08B

## Files

| File | Description |
|------|-------------|
| `model.py` | Model architecture (Diff-Llama-MTP) |
| `train.py` | Training script with WSD scheduler |
| `inference.py` | Generation with KV cache |
| `train_colab.ipynb` | Google Colab notebook |

## Quick Start (Colab TPU v6e-1)

```python
# Run training
!python train.py

# The script auto-configures:
# - Learning rate: scaled by model size
# - Total steps: computed from data size
# - WSD schedule: 90% stable, 10% decay
```

## Training Strategy

- **Multi-Epoch**: 4 epochs with dropout (0.1) for degradation protection
- **Checkpointing**: Every epoch for offline HumanEval evaluation
- **Evaluation**: Use `eval_humaneval.py` to score each epoch on HumanEval (pass@1/ pass@k)
- **Data**: 24B tokens preprocessed in Parquet format

## References

- [Differential Transformer (ICLR 2025)](https://arxiv.org/abs/2410.05258)
- [Multi-Token Prediction (Meta)](https://arxiv.org/abs/2404.19737)
- [MiniCPM WSD Schedule](https://arxiv.org/abs/2404.06395)
- [Qwen2.5-Coder](https://arxiv.org/abs/2409.12186)
- [To Repeat or Not To Repeat (NeurIPS 2023)](https://arxiv.org/abs/2305.13230)

## License

MIT
