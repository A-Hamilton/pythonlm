"""PythonCoder Training v2 - Minimal Build for TPU v6e-1 (January 2026)

Simplified training script for Diff-Llama-MTP architecture.
Removed MoE routing, Mamba layers, and other complexity.

Target: Train 1.1B model on TPU v6e-1 with:
- Differential Attention + GQA (65% parameter efficiency)
- Multi-Token Prediction (MTP=1, +12-17% HumanEval)
- YaRN (4x context extension, train at 2K, infer at 8K)
- WSD Scheduler (90% stable phase, flexible training)

Automated Scaling:
- LR: Computed from model size (smaller models use higher LR)
- Steps: Computed from data size (3 data epochs, ~72 tokens/param)
- Sources: MiniCPM (arXiv:2404.06395), Qwen2.5-Coder (arXiv:2409.12186)
"""

import os
import time
import glob
import shutil
from pathlib import Path
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import Mesh

# Import model
from model import PythonCoderModel, CONFIG_1B, SPECIAL_TOKENS, create_model


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Training config for 1.1B model on TPU v6e-1 with WSD scheduler.

    Multi-Epoch Training Strategy (Research-Backed):
    ================================================
    Sources:
    - "To Repeat or Not To Repeat" (NeurIPS 2023): https://arxiv.org/abs/2305.13230
    - MiniCPM WSD: https://arxiv.org/abs/2404.06395
    - Galactica: Trained 4.25 epochs successfully

    Key findings:
    1. 4 epochs is generally safe before degradation
    2. Dropout is the ONLY effective regularization (apply after epoch 2-3)
    3. Checkpoint frequently during stable phase
    4. Validation loss is unreliable - use downstream eval (HumanEval)
    5. WSD allows testing different decay branches from same checkpoint
    """

    # Hardware
    tpu_version: str = "v6e-1"

    # Optimization for 1.1B model:
    #   Static (params + optimizer): ~8.6 GB
    #   Vocab gradients (batch=8): ~14 GB
    #   Other activations: ~4 GB
    #   Total: ~26.6 GB < 32 GB (5.4 GB headroom)
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 8  # Effective batch = 64 (8 * 8)
    learning_rate: float = 5e-4  # Aggressive for 1B (matches Qwen/Phi scaling)
    weight_decay: float = 0.1
    warmup_steps: int = 2000  # Standard warmup
    stable_ratio: float = 0.9  # WSD: 90% stable, 10% decay
    grad_clip: float = 1.0

    # Data
    max_seq_len: int = 2048  # Start with 2048, can increase

    # Training - Multi-Epoch Strategy
    steps_per_epoch: int = 5000  # ~2.4% of 1 data epoch
    log_every: int = 50
    max_epochs: int = 100  # High ceiling - rely on checkpointing
    checkpoint_every_epoch: int = 1  # Save every epoch for analysis

    # Multi-epoch degradation detection
    # NOTE: Validation loss is unreliable for overfitting detection!
    # Best practice: Run HumanEval on checkpoints offline
    early_stopping_patience: int = 10  # Increased - loss may plateau then improve
    min_improvement: float = 0.001  # Loosened - small improvements still valuable

    # Dropout is enabled in model_v2.py (dropout_prob=0.1)
    # Research: "Dropout alone is highly effective" for multi-epoch degradation
    # Source: "To Repeat or Not To Repeat" (NeurIPS 2023)

    # Paths
    base_dir: str = "/content/drive/MyDrive/python-coder-v6e"

    def __post_init__(self):
        self.data_dir = f"{self.base_dir}/preprocessed_data"
        self.checkpoint_dir = f"{self.base_dir}/checkpoints_v2"
        self.effective_batch_size = self.micro_batch_size * self.gradient_accumulation_steps


CONFIG = TrainingConfig()


# =============================================================================
# Automated Scaling (Benchmark-Justified)
# =============================================================================

def compute_training_config(num_params: int, total_tokens: int, target_epochs: int = 4) -> dict:
    """Auto-compute training hyperparameters from model/data size.

    Multi-Epoch Strategy (Research-Backed):
    - 4 epochs: Generally safe (Galactica trained 4.25 epochs successfully)
    - Beyond 4: Apply dropout starting epoch 3 to prevent degradation
    - Checkpoint every epoch for offline HumanEval evaluation

    Sources:
    - LR: Qwen/DeepSeek empirical (smaller models use higher LR)
      https://arxiv.org/abs/2409.12186 (Qwen2.5-Coder uses 3e-4)
    - Multi-epoch: "To Repeat or Not To Repeat" (NeurIPS 2023)
      https://arxiv.org/abs/2305.13230
    - WSD: MiniCPM 90% stable phase
      https://arxiv.org/abs/2404.06395

    Args:
        num_params: Model parameter count (e.g., 1.08e9)
        total_tokens: Total training tokens available (e.g., 24e9)
        target_epochs: Number of data epochs to train (default: 4)

    Returns:
        dict with training configuration
    """
    seq_len = CONFIG.max_seq_len  # 2048
    batch_size = CONFIG.effective_batch_size  # 64
    tokens_per_step = seq_len * batch_size  # 131,072

    # Calculate steps for target epochs
    steps_per_data_epoch = int(total_tokens / tokens_per_step)
    total_steps = steps_per_data_epoch * target_epochs

    # LR: scale up for smaller models (empirical formula)
    # Qwen2.5-Coder-1.5B uses 3e-4, smaller models benefit from higher LR
    base_lr = 3e-4  # Qwen baseline for 1.5B
    reference_size = 1.5e9
    size_factor = (reference_size / num_params) ** 0.5
    learning_rate = base_lr * size_factor
    learning_rate = min(max(learning_rate, 1e-4), 1e-3)  # Clamp to safe range

    # WSD boundaries (MiniCPM standard)
    warmup_steps = 2000
    stable_ratio = 0.9

    # Compute tokens per parameter
    tokens_per_param = (total_tokens * target_epochs) / num_params

    # Chinchilla assessment
    chinchilla_ratio = total_tokens / num_params  # Single epoch
    chinchilla_status = "optimal" if 15 <= chinchilla_ratio <= 25 else (
        "under-trained" if chinchilla_ratio < 15 else "over-trained (multi-epoch needed)"
    )

    return {
        "total_steps": total_steps,
        "steps_per_data_epoch": steps_per_data_epoch,
        "target_epochs": target_epochs,
        "learning_rate": round(learning_rate, 6),
        "warmup_steps": warmup_steps,
        "stable_ratio": stable_ratio,
        "tokens_per_param": round(tokens_per_param, 1),
        "chinchilla_ratio": round(chinchilla_ratio, 1),
        "chinchilla_status": chinchilla_status,
        "effective_batch_size": batch_size,
        "tokens_per_step": tokens_per_step,
    }


# =============================================================================
# Environment Setup
# =============================================================================

def setup_environment():
    """Setup TPU environment."""
    # Mount Drive (Colab)
    if os.path.exists("/content"):
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
        except Exception:
            pass

    os.makedirs(CONFIG.checkpoint_dir, exist_ok=True)

    # JAX cache
    cache_dir = f"{CONFIG.base_dir}/jax_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

    # Setup mesh
    devices = jax.devices()
    print(f"JAX Devices: {len(devices)}x {devices[0].platform}")
    mesh = Mesh(devices, axis_names=('fsdp',))

    return mesh


# =============================================================================
# Data Loading
# =============================================================================

def create_data_iterator(mesh: Mesh):
    """Create streaming data iterator."""
    import grain.python as grain

    shard_paths = sorted(glob.glob(f"{CONFIG.data_dir}/train/*.parquet"))
    if not shard_paths:
        shard_paths = sorted(glob.glob("./preprocessed_data/train/*.parquet"))

    if not shard_paths:
        raise ValueError(f"No data found in {CONFIG.data_dir}/train/")

    print(f"Found {len(shard_paths)} data shards")

    try:
        import grain.experimental as grain_exp

        filenames_ds = grain.MapDataset.source(shard_paths)
        parquet_ds = filenames_ds.map(grain_exp.ParquetIterDataset)
        interleaved = grain_exp.InterleaveIterDataset(parquet_ds, cycle_length=min(len(shard_paths), 4))
        shuffled = grain_exp.WindowShuffleIterDataset(interleaved, window_size=10000, seed=42)

        def transform(row):
            input_ids = row['input_ids']
            labels = row['labels']
            if hasattr(input_ids, 'tolist'):
                input_ids = input_ids.tolist()
            if hasattr(labels, 'tolist'):
                labels = labels.tolist()
            return {'input_ids': input_ids, 'labels': labels}

        dataset = shuffled.map(transform)
        batched = dataset.batch(CONFIG.micro_batch_size, drop_remainder=True)
        return iter(batched)

    except Exception as e:
        print(f"Grain failed: {e}, using basic loader")
        import pyarrow.parquet as pq

        def basic_iterator():
            while True:
                for path in shard_paths:
                    table = pq.read_table(path)
                    for i in range(0, len(table), CONFIG.micro_batch_size):
                        batch = table.slice(i, CONFIG.micro_batch_size)
                        if len(batch) < CONFIG.micro_batch_size:
                            continue
                        yield {
                            'input_ids': batch['input_ids'].to_pylist(),
                            'labels': batch['labels'].to_pylist()
                        }
        return basic_iterator()


# =============================================================================
# Optimizer
# =============================================================================

def create_optimizer(model: PythonCoderModel, total_steps: int) -> nnx.Optimizer:
    """Create optimizer with WSD (Warmup-Stable-Decay) schedule.

    WSD allows training for any number of epochs without degradation:
    - Warmup: Linear ramp from 1% to 100% of LR
    - Stable: Constant LR for main training (90% of steps)
    - Decay: Cosine decay to 10% in final phase

    Unlike cosine decay which forces early LR death, WSD keeps learning
    rate high until you explicitly decide to finish training.
    """
    # Calculate phase boundaries
    decay_start = int(total_steps * CONFIG.stable_ratio)
    decay_steps = total_steps - decay_start

    # Phase 1: Warmup (linear ramp)
    warmup_schedule = optax.linear_schedule(
        init_value=CONFIG.learning_rate * 0.01,
        end_value=CONFIG.learning_rate,
        transition_steps=CONFIG.warmup_steps
    )

    # Phase 2: Stable (constant LR - the key innovation)
    stable_schedule = optax.constant_schedule(CONFIG.learning_rate)

    # Phase 3: Decay (cosine cooldown)
    decay_schedule = optax.cosine_decay_schedule(
        init_value=CONFIG.learning_rate,
        decay_steps=decay_steps,
        alpha=0.1  # End at 10% of peak
    )

    # Combine phases
    schedule = optax.join_schedules(
        schedules=[warmup_schedule, stable_schedule, decay_schedule],
        boundaries=[CONFIG.warmup_steps, decay_start]
    )

    tx = optax.chain(
        optax.clip_by_global_norm(CONFIG.grad_clip),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.95,
            weight_decay=CONFIG.weight_decay,
            mu_dtype=jnp.bfloat16,
        ),
    )

    return nnx.Optimizer(model, tx, wrt=nnx.Param)


# =============================================================================
# Training Step (Fixed: Proper Gradient Accumulation + Dropout)
# =============================================================================

@nnx.jit
def compute_grads(model: PythonCoderModel, batch: dict):
    """Compute gradients for a single micro-batch (no optimizer update).

    Bug fixes:
    1. Dropout: Pass deterministic=False to enable dropout during training
       Source: https://flax.readthedocs.io/en/latest/nnx_basics.html
    2. Gradient accumulation: Only compute grads, don't update optimizer
       Source: https://optax.readthedocs.io/en/latest/_collections/examples/gradient_accumulation.html
    """
    def loss_fn(model):
        # CRITICAL: deterministic=False enables dropout during training
        output = model(batch['input_ids'], labels=batch['labels'], deterministic=False)
        return output['loss']

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    return loss, grads


@nnx.jit
def apply_grads(model: PythonCoderModel, optimizer: nnx.Optimizer, grads: dict):
    """Apply accumulated gradients to model."""
    optimizer.update(model, grads)


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    model: PythonCoderModel,
    optimizer: nnx.Optimizer,
    epoch: int,
    loss: float,
    global_step: int = 0,
    data_epoch: int = 1
):
    """Save checkpoint with metadata for multi-epoch analysis.

    Checkpoints are saved every epoch to enable:
    - Offline HumanEval evaluation (validation loss unreliable)
    - Testing different WSD decay branches from same checkpoint
    - Finding best checkpoint per data epoch
    """
    ckpt_path = Path(CONFIG.checkpoint_dir) / f"epoch_{epoch}"
    if ckpt_path.exists():
        shutil.rmtree(ckpt_path)

    state = {
        'model': nnx.state(model),
        'optimizer': nnx.state(optimizer),
        'epoch': epoch,
        'loss': loss,
        'global_step': global_step,
        'data_epoch': data_epoch,
    }

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(ckpt_path, state)
    checkpointer.wait_until_finished()
    print(f"Saved: {ckpt_path} (data_epoch={data_epoch}, step={global_step:,})")


def load_checkpoint(model: PythonCoderModel, optimizer: nnx.Optimizer) -> tuple:
    """Load latest checkpoint if exists.

    Returns:
        tuple: (epoch, global_step, data_epoch) or (0, 0, 1) if no checkpoint
    """
    ckpt_dir = Path(CONFIG.checkpoint_dir)
    if not ckpt_dir.exists():
        return 0, 0, 1

    checkpoints = list(ckpt_dir.glob("epoch_*"))
    if not checkpoints:
        return 0, 0, 1

    latest = max(checkpoints, key=lambda p: int(p.name.split("_")[1]))
    print(f"Loading: {latest}")

    model_state = nnx.state(model)
    opt_state = nnx.state(optimizer)

    abstract_state = {
        'model': jax.tree.map(ocp.utils.to_shape_dtype_struct, model_state),
        'optimizer': jax.tree.map(ocp.utils.to_shape_dtype_struct, opt_state),
        'epoch': 0,
        'loss': 0.0,
        'global_step': 0,
        'data_epoch': 1,
    }

    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(latest, abstract_state)

    nnx.update(model, restored['model'])
    nnx.update(optimizer, restored['optimizer'])

    # Handle old checkpoints without new fields
    global_step = restored.get('global_step', restored['epoch'] * CONFIG.steps_per_epoch)
    data_epoch = restored.get('data_epoch', 1)

    print(f"Resumed: epoch={restored['epoch']}, step={global_step:,}, data_epoch={data_epoch}")
    return restored['epoch'], global_step, data_epoch


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    print("=" * 60)
    print("PythonCoder v2 Training - Diff-Llama-MTP")
    print("=" * 60)
    print(f"Config: {CONFIG_1B.num_layers}L, {CONFIG_1B.hidden_size}H")
    print(f"Features: DiffAttn + MTP + YaRN")
    print(f"Batch: {CONFIG.micro_batch_size} x {CONFIG.gradient_accumulation_steps} = {CONFIG.effective_batch_size}")
    print("=" * 60)

    # Setup
    mesh = setup_environment()

    # Create model
    print("\nCreating model...")
    rngs = nnx.Rngs(42)
    model = create_model(CONFIG_1B, rngs=rngs)

    params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model)))
    print(f"Parameters: {params:,} ({params/1e9:.2f}B)")

    # Compute automated training config
    # Data: ~24B tokens (45GB parquet, 2390 shards)
    TOTAL_TOKENS = 24_000_000_000  # 24B tokens
    TARGET_EPOCHS = 4  # Research: 4 epochs safe, beyond needs dropout
    auto_config = compute_training_config(
        num_params=params,
        total_tokens=TOTAL_TOKENS,
        target_epochs=TARGET_EPOCHS
    )

    CONFIG.learning_rate = auto_config['learning_rate']
    CONFIG.warmup_steps = auto_config['warmup_steps']
    CONFIG.stable_ratio = auto_config['stable_ratio']

    print("\n" + "─" * 60)
    print("AUTOMATED SCALING CONFIG (Benchmark-Justified)")
    print("─" * 60)
    print(f"  Data Tokens:      {TOTAL_TOKENS/1e9:.1f}B")
    print(f"  Target Epochs:    {auto_config['target_epochs']} data epochs")
    print(f"  Steps/Epoch:      {auto_config['steps_per_data_epoch']:,}")
    print(f"  Total Steps:      {auto_config['total_steps']:,}")
    print(f"  Learning Rate:    {CONFIG.learning_rate:.2e}")
    print(f"  Tokens/Param:     {auto_config['tokens_per_param']:.1f}x total")
    print(f"  Chinchilla:       {auto_config['chinchilla_ratio']:.1f}x ({auto_config['chinchilla_status']})")
    print("─" * 60)
    print("MULTI-EPOCH STRATEGY (Research: arxiv.org/abs/2305.13230)")
    print("─" * 60)
    print(f"  Dropout:          Enabled (rate=0.1 in model)")
    print(f"  Checkpoints:      Every epoch (for offline HumanEval)")
    print(f"  Early Stop:       Patience={CONFIG.early_stopping_patience} epochs")
    print("  NOTE: Validation loss unreliable - use downstream eval!")
    print("─" * 60)

    # Use computed total_steps, keep hardcoded LR (already tuned)
    total_steps = auto_config['total_steps']
    optimizer = create_optimizer(model, total_steps)

    # Load checkpoint
    start_epoch, start_global_step, start_data_epoch = load_checkpoint(model, optimizer)

    # Load data
    print("\nLoading data...")
    data_iter = create_data_iterator(mesh)

    # Training loop
    print("\nStarting training...")
    best_loss = float('inf')
    patience = 0
    global_step = start_global_step  # Resume from checkpoint
    steps_per_data_epoch = auto_config['steps_per_data_epoch']

    # Track best checkpoints for each data epoch
    best_checkpoints = {}

    for epoch in range(start_epoch + 1, CONFIG.max_epochs + 1):
        epoch_start = time.time()
        epoch_losses = []

        # Calculate which data epoch we're in (1-indexed)
        data_epoch = (global_step // steps_per_data_epoch) + 1
        progress_in_data_epoch = (global_step % steps_per_data_epoch) / steps_per_data_epoch * 100

        print(f"\n{'─' * 60}")
        print(f"EPOCH {epoch} | Data Epoch {data_epoch} ({progress_in_data_epoch:.1f}% through)")
        print(f"{'─' * 60}")

        for step in range(1, CONFIG.steps_per_epoch + 1):
            accumulated_loss = 0.0
            accumulated_grads = None

            # Accumulate gradients over micro-batches (FIX: don't update optimizer per micro-batch)
            for micro_step in range(CONFIG.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = create_data_iterator(mesh)
                    batch = next(data_iter)

                batch = {
                    'input_ids': jnp.array(batch['input_ids'])[:, :CONFIG.max_seq_len],
                    'labels': jnp.array(batch['labels'])[:, :CONFIG.max_seq_len],
                }

                # Compute gradients without updating (FIX)
                loss, grads = compute_grads(model, batch)
                accumulated_loss += float(loss)

                # Accumulate gradients
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = jax.tree.map(
                        lambda a, g: a + g, accumulated_grads, grads
                    )

            # Average gradients and apply once (FIX: was applying 8x per step before)
            scale = 1.0 / CONFIG.gradient_accumulation_steps
            averaged_grads = jax.tree.map(lambda g: g * scale, accumulated_grads)
            apply_grads(model, optimizer, averaged_grads)

            loss = accumulated_loss / CONFIG.gradient_accumulation_steps
            epoch_losses.append(loss)
            global_step += 1

            if step % CONFIG.log_every == 0:
                avg_loss = sum(epoch_losses[-CONFIG.log_every:]) / min(len(epoch_losses), CONFIG.log_every)
                current_data_epoch = (global_step // steps_per_data_epoch) + 1
                print(f"Step {step:5d}/{CONFIG.steps_per_epoch} | Loss: {avg_loss:.4f} | DataEpoch: {current_data_epoch}")

        # Epoch summary
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_time = time.time() - epoch_start
        final_data_epoch = (global_step // steps_per_data_epoch) + 1

        print(f"\nEpoch {epoch} Complete")
        print(f"  Loss: {epoch_loss:.4f}")
        print(f"  Data Epoch: {final_data_epoch}")
        print(f"  Global Steps: {global_step:,}")
        print(f"  Time: {epoch_time:.1f}s")

        # Save checkpoint with metadata
        save_checkpoint(model, optimizer, epoch, epoch_loss, global_step, final_data_epoch)

        # Track best loss per data epoch
        if final_data_epoch not in best_checkpoints or epoch_loss < best_checkpoints[final_data_epoch]['loss']:
            best_checkpoints[final_data_epoch] = {'epoch': epoch, 'loss': epoch_loss}
            print(f"  [NEW BEST for Data Epoch {final_data_epoch}]")

        # Early stopping (loosened - validation loss unreliable)
        if epoch_loss < best_loss - CONFIG.min_improvement:
            best_loss = epoch_loss
            patience = 0
        else:
            patience += 1
            if patience >= CONFIG.early_stopping_patience:
                print(f"\nEarly stopping: No improvement for {CONFIG.early_stopping_patience} epochs")
                print("NOTE: Consider running HumanEval on checkpoints - loss may not reflect quality!")
                break

        # Info about multi-epoch progress
        if final_data_epoch >= 4:
            print(f"  [Data Epoch {final_data_epoch}] Dropout active - multi-epoch protection enabled")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Final Data Epoch: {final_data_epoch}")
    print("=" * 60)
    print("\nBEST CHECKPOINTS BY DATA EPOCH:")
    print("(Run HumanEval on each to find true best)")
    for de, info in sorted(best_checkpoints.items()):
        print(f"  Data Epoch {de}: Epoch {info['epoch']} (loss={info['loss']:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
