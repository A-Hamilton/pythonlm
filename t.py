#!/usr/bin/env python3
"""
JAX Windows Diagnostic Script
Tests if JAX is installed and which backends are available.
"""

import sys
import platform

print("=" * 50)
print("JAX WINDOWS DIAGNOSTIC")
print("=" * 50)

# System info
print(f"\nSystem: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")

# Check JAX import
print("\n[1] Checking JAX import...")
try:
    import jax
    print(f"    ✓ JAX version: {jax.__version__}")
except ImportError as e:
    print(f"    ✗ JAX not installed: {e}")
    print("\n    To install JAX (CPU only):")
    print("    pip install jax")
    sys.exit(1)

# Check available backends
print("\n[2] Checking backends...")
try:
    backends = jax.lib.xla_bridge.get_backend().platform
    print(f"    Default backend: {backends}")
except Exception as e:
    print(f"    Backend error: {e}")

# List all devices
print("\n[3] Available devices:")
try:
    devices = jax.devices()
    for d in devices:
        print(f"    - {d}")
    
    if len(devices) == 0:
        print("    (no devices found)")
except Exception as e:
    print(f"    Error listing devices: {e}")

# Check for GPU/TPU
print("\n[4] GPU/TPU check:")
try:
    gpu_devices = jax.devices('gpu')
    print(f"    ✓ GPU devices: {len(gpu_devices)}")
    for d in gpu_devices:
        print(f"      - {d}")
except RuntimeError:
    print("    ✗ No GPU backend available")

try:
    tpu_devices = jax.devices('tpu')
    print(f"    ✓ TPU devices: {len(tpu_devices)}")
except RuntimeError:
    print("    ✗ No TPU backend available")

# Test basic computation
print("\n[5] Testing basic computation...")
try:
    import jax.numpy as jnp
    
    # Simple array operation
    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x)
    result = float(y[0, 0])
    
    print(f"    ✓ Matrix multiply works (1000x1000 @ 1000x1000)")
    print(f"    Result[0,0] = {result} (expected: 1000.0)")
except Exception as e:
    print(f"    ✗ Computation failed: {e}")

# Test JIT compilation
print("\n[6] Testing JIT compilation...")
try:
    @jax.jit
    def simple_fn(x):
        return x * 2 + 1
    
    result = simple_fn(jnp.array([1.0, 2.0, 3.0]))
    print(f"    ✓ JIT works: [1,2,3] * 2 + 1 = {result}")
except Exception as e:
    print(f"    ✗ JIT failed: {e}")

# Test gradient
print("\n[7] Testing autodiff...")
try:
    def loss_fn(x):
        return jnp.sum(x ** 2)
    
    grad_fn = jax.grad(loss_fn)
    x = jnp.array([1.0, 2.0, 3.0])
    grads = grad_fn(x)
    print(f"    ✓ Autodiff works: grad(sum(x²)) at [1,2,3] = {grads}")
except Exception as e:
    print(f"    ✗ Autodiff failed: {e}")

# Summary
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)

backend = jax.devices()[0].platform if jax.devices() else "none"
print(f"Backend: {backend}")
