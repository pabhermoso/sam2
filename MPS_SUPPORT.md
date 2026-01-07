# SAM2 MPS (Apple Silicon GPU) Support

This document describes the changes made to enable SAM2 to run on Apple Silicon Macs using MPS (Metal Performance Shaders) backend.

## Summary

SAM2 can now run on MPS devices (M1/M2/M3/M4 Macs) without requiring `SAM2_DEMO_FORCE_CPU_DEVICE=1`. Testing shows approximately **2x speedup** compared to CPU on M4 Mac.

## Changes Made

### 1. `sam2/modeling/sam/transformer.py`

**Issue:** The `RoPEAttention` class hardcoded CUDA device for `freqs_cis` initialization.

**Fix:** Added MPS device support in the initialization:

```python
# Before:
self.freqs_cis = freqs_cis.to("cuda") if torch.cuda.is_available() else freqs_cis

# After:
if torch.cuda.is_available():
    self.freqs_cis = freqs_cis.to("cuda")
elif torch.backends.mps.is_available():
    self.freqs_cis = freqs_cis.to("mps")
else:
    self.freqs_cis = freqs_cis
```

### 2. `sam2/modeling/position_encoding.py`

**Issue:** Cache warmup only ran on CUDA devices, and complex tensor repeat operations were restricted to CUDA.

**Fix:** 
- Extended cache warmup to include MPS devices
- Added MPS device detection for complex tensor repeat operations

### 3. `demo/backend/server/inference/image_segmentor.py`

**Issue:** `autocast_context()` only supported CUDA, returning null context for MPS.

**Fix:** Added MPS autocast support with float16 (bfloat16 is not fully supported on MPS):

```python
def autocast_context(self):
    if self.device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    elif self.device.type == "mps":
        return torch.autocast("mps", dtype=torch.float16)
    return contextlib.nullcontext()
```

### 4. `demo/backend/server/inference/predictor.py`

**Issue:** Same autocast issue as `image_segmentor.py`, plus CUDA memory stats functions called on MPS.

**Fix:**
- Added MPS autocast support
- Made GPU memory stats device-aware

## Benchmark Results (M4 Mac)

### Tiny Model (1200x1800 image)

| Device | set_image | predict | total |
|--------|-----------|---------|-------|
| CPU    | 0.711s    | 0.031s  | 0.742s |
| MPS    | 0.349s    | 0.021s  | 0.370s |
| **Speedup** | 2.0x | 1.5x | **2.0x** |

### Base Plus Model (1024x1024 image)

| Device | Total Time |
|--------|------------|
| MPS    | ~0.78s     |

## Running the Benchmark

```bash
cd /Users/phm/Downloads/VIRTUAL_TRY_ON/sam2
conda activate sam2-demo
python test_mps_benchmark.py
```

## Running the Demo Server on MPS

The server will now automatically use MPS if available:

```bash
cd demo/backend/server
APP_ROOT="$(pwd)/../../../" \
API_URL=http://localhost:7263 \
MODEL_SIZE=tiny \
DATA_PATH="$(pwd)/../../data" \
DEFAULT_VIDEO_PATH=gallery/05_default_juggle.mp4 \
gunicorn \
    --worker-class gthread app:app \
    --workers 1 \
    --threads 2 \
    --bind 0.0.0.0:7263 \
    --timeout 60
```

You can still force CPU if needed:

```bash
SAM2_DEMO_FORCE_CPU_DEVICE=1 <same command>
```

## Known Limitations

1. **Gunicorn/Forked Workers (CRITICAL):** MPS (Metal) **does not work** in forked processes on macOS. This means:
   - Running under gunicorn with workers will crash with `MTLCompilerService` errors
   - The code now auto-detects gunicorn and falls back to CPU
   - For standalone Python scripts or Flask dev server, MPS works fine
   - If you need MPS acceleration, use `python app.py` directly instead of gunicorn

2. **Performance Variance:** MPS performance can vary depending on the specific operations. Some tensor operations may be slower on MPS than CPU (see benchmark for raw matmul results).

3. **Memory Management:** For video processing, frames are offloaded to CPU to avoid MPS memory fragmentation (this is already handled in the demo code).

4. **Numerical Precision:** MPS uses float16 autocast instead of bfloat16 (CUDA). Results should be numerically similar but may have slight differences.

5. **CUDA Extensions:** The connected components CUDA extension (`sam2._C`) doesn't work on MPS, but the code already has fallback handling for this.

## Gunicorn Workaround

The Metal framework doesn't work correctly in forked processes. When running under gunicorn:

```bash
# This will CRASH with MPS:
gunicorn --worker-class gthread app:app --workers 1

# Safe option 1: Force CPU mode
SAM2_DEMO_FORCE_CPU_DEVICE=1 gunicorn --worker-class gthread app:app --workers 1

# Safe option 2: Use Flask dev server (supports MPS)
python -c "from app import app; app.run(host='0.0.0.0', port=7263, threaded=True)"
```

The code now automatically detects gunicorn workers and falls back to CPU to prevent crashes.

## PyTorch Version

Tested with PyTorch 2.9.1 on macOS with MPS backend.

