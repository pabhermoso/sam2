#!/usr/bin/env python3
"""
MPS vs CPU Benchmark for SAM2

This script benchmarks SAM2 image segmentation on:
1. CPU (baseline)
2. MPS (Apple Silicon GPU)

Run with:
    cd /Users/phm/Downloads/VIRTUAL_TRY_ON/sam2
    python test_mps_benchmark.py
"""

import os
import sys
import time
import warnings
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image

# Suppress hydra's verbose logging
import logging
logging.getLogger().setLevel(logging.ERROR)

# Add sam2 to path if not installed
SAM2_ROOT = Path(__file__).parent
sys.path.insert(0, str(SAM2_ROOT))


def check_mps_availability():
    """Check if MPS is available and working."""
    print("=" * 60)
    print("MPS AVAILABILITY CHECK")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS backend available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    if torch.backends.mps.is_available():
        try:
            # Test basic MPS operation
            x = torch.randn(10, 10, device="mps")
            y = x @ x.T
            torch.mps.synchronize()
            print("MPS basic test: PASSED ✓")
            return True
        except Exception as e:
            print(f"MPS basic test: FAILED ✗ - {e}")
            return False
    else:
        print("MPS is not available on this system")
        return False


def get_test_image():
    """Get or create a test image."""
    # Try to find an existing image in the notebooks folder
    test_images = [
        SAM2_ROOT / "notebooks" / "images" / "truck.jpg",
        SAM2_ROOT / "notebooks" / "images" / "groceries.jpg",
        SAM2_ROOT / "notebooks" / "images" / "cars.jpg",
    ]
    
    for img_path in test_images:
        if img_path.exists():
            return np.array(Image.open(img_path).convert("RGB"))
    
    # Create a synthetic test image
    print("Creating synthetic test image (no sample images found)")
    return np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)


def get_checkpoint_and_config(model_size="tiny"):
    """Get checkpoint path and config for SAM2 model."""
    checkpoints_dir = SAM2_ROOT / "checkpoints"
    
    configs = {
        "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
        "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
        "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
        "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
    }
    
    config_file, ckpt_name = configs.get(model_size, configs["tiny"])
    ckpt_path = checkpoints_dir / ckpt_name
    
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Please download checkpoints first using: cd checkpoints && ./download_ckpts.sh")
        sys.exit(1)
    
    return config_file, str(ckpt_path)


@contextmanager
def timer(description: str):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  {description}: {elapsed:.4f}s")
    return elapsed


def benchmark_device(device_name: str, num_warmup: int = 2, num_runs: int = 5, model_size: str = "tiny"):
    """Benchmark SAM2 on a specific device."""
    print(f"\n{'=' * 60}")
    print(f"BENCHMARKING ON: {device_name.upper()}")
    print(f"{'=' * 60}")
    
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    
    config_file, ckpt_path = get_checkpoint_and_config(model_size)
    
    device = torch.device(device_name)
    
    # Build model
    print(f"\n1. Building SAM2 model ({model_size})...")
    try:
        model_start = time.perf_counter()
        model = build_sam2(
            config_file=config_file,
            ckpt_path=ckpt_path,
            device=device,
            mode="eval",
        )
        model.to(device)
        model_time = time.perf_counter() - model_start
        print(f"   Model built in {model_time:.2f}s")
    except Exception as e:
        print(f"   FAILED to build model: {e}")
        return None
    
    predictor = SAM2ImagePredictor(model)
    
    # Get test image
    print("\n2. Loading test image...")
    image = get_test_image()
    print(f"   Image shape: {image.shape}")
    
    # Warmup runs
    print(f"\n3. Warmup ({num_warmup} runs)...")
    for i in range(num_warmup):
        try:
            predictor.set_image(image)
            # Sample point in center of image
            point_coords = np.array([[image.shape[1] // 2, image.shape[0] // 2]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int64)
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
            if device_name == "mps":
                torch.mps.synchronize()
            print(f"   Warmup {i+1}: OK")
        except Exception as e:
            print(f"   Warmup {i+1} FAILED: {e}")
            return None
    
    # Benchmark runs
    print(f"\n4. Benchmark runs ({num_runs} runs)...")
    
    set_image_times = []
    predict_times = []
    total_times = []
    
    for i in range(num_runs):
        # Time set_image
        start = time.perf_counter()
        predictor.set_image(image)
        if device_name == "mps":
            torch.mps.synchronize()
        set_image_time = time.perf_counter() - start
        set_image_times.append(set_image_time)
        
        # Time predict
        point_coords = np.array([[image.shape[1] // 2, image.shape[0] // 2]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int64)
        
        start = time.perf_counter()
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )
        if device_name == "mps":
            torch.mps.synchronize()
        predict_time = time.perf_counter() - start
        predict_times.append(predict_time)
        
        total_time = set_image_time + predict_time
        total_times.append(total_time)
        
        print(f"   Run {i+1}: set_image={set_image_time:.4f}s, predict={predict_time:.4f}s, total={total_time:.4f}s")
    
    # Calculate statistics
    results = {
        "device": device_name,
        "model_size": model_size,
        "set_image_mean": np.mean(set_image_times),
        "set_image_std": np.std(set_image_times),
        "predict_mean": np.mean(predict_times),
        "predict_std": np.std(predict_times),
        "total_mean": np.mean(total_times),
        "total_std": np.std(total_times),
        "mask_shape": masks[0].shape if masks is not None else None,
    }
    
    print(f"\n5. Results Summary:")
    print(f"   set_image: {results['set_image_mean']:.4f}s ± {results['set_image_std']:.4f}s")
    print(f"   predict:   {results['predict_mean']:.4f}s ± {results['predict_std']:.4f}s")
    print(f"   total:     {results['total_mean']:.4f}s ± {results['total_std']:.4f}s")
    print(f"   mask shape: {results['mask_shape']}")
    
    # Clean up
    del predictor
    del model
    if device_name == "mps":
        torch.mps.empty_cache()
    elif device_name == "cuda":
        torch.cuda.empty_cache()
    
    return results


def test_tensor_operations():
    """Test basic tensor operations on MPS vs CPU."""
    print("\n" + "=" * 60)
    print("BASIC TENSOR OPERATIONS TEST")
    print("=" * 60)
    
    sizes = [(512, 512), (1024, 1024), (2048, 2048)]
    
    for size in sizes:
        print(f"\nMatrix size: {size[0]}x{size[1]}")
        
        # CPU
        x_cpu = torch.randn(*size, dtype=torch.float32, device="cpu")
        start = time.perf_counter()
        _ = x_cpu @ x_cpu.T
        cpu_time = time.perf_counter() - start
        print(f"  CPU: {cpu_time:.4f}s")
        
        # MPS
        if torch.backends.mps.is_available():
            x_mps = torch.randn(*size, dtype=torch.float32, device="mps")
            start = time.perf_counter()
            _ = x_mps @ x_mps.T
            torch.mps.synchronize()
            mps_time = time.perf_counter() - start
            print(f"  MPS: {mps_time:.4f}s")
            print(f"  Speedup: {cpu_time/mps_time:.2f}x")


def main():
    print("\n" + "=" * 60)
    print("SAM2 MPS BENCHMARK")
    print("=" * 60)
    
    # Check MPS availability
    mps_available = check_mps_availability()
    
    # Test basic tensor operations first
    if mps_available:
        test_tensor_operations()
    
    # Benchmark SAM2
    print("\n\n" + "=" * 60)
    print("SAM2 MODEL BENCHMARK")
    print("=" * 60)
    
    model_size = "tiny"  # Use tiny for faster testing; change to "base_plus" for better accuracy
    num_warmup = 2
    num_runs = 3
    
    results = {}
    
    # Always benchmark CPU as baseline
    results["cpu"] = benchmark_device("cpu", num_warmup, num_runs, model_size)
    
    # Benchmark MPS if available
    if mps_available:
        results["mps"] = benchmark_device("mps", num_warmup, num_runs, model_size)
    
    # Summary comparison
    print("\n\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    
    if results["cpu"] and results.get("mps"):
        cpu_time = results["cpu"]["total_mean"]
        mps_time = results["mps"]["total_mean"]
        speedup = cpu_time / mps_time
        
        print(f"\nCPU average time:  {cpu_time:.4f}s")
        print(f"MPS average time:  {mps_time:.4f}s")
        print(f"Speedup (MPS vs CPU): {speedup:.2f}x")
        
        if speedup > 1:
            print(f"\n✓ MPS is {speedup:.2f}x FASTER than CPU!")
        elif speedup < 1:
            print(f"\n✗ MPS is {1/speedup:.2f}x SLOWER than CPU")
        else:
            print("\n~ MPS and CPU are approximately equal")
    elif results["cpu"] and not mps_available:
        print("\nMPS not available - only CPU results shown")
        print(f"CPU average time: {results['cpu']['total_mean']:.4f}s")
    elif results["cpu"] and not results.get("mps"):
        print("\nMPS benchmark failed - only CPU results shown")
        print(f"CPU average time: {results['cpu']['total_mean']:.4f}s")
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

