import base64
import contextlib
import io
import logging
import os
from threading import Lock
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from app_conf import APP_ROOT, MODEL_SIZE
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

logger = logging.getLogger(__name__)


def _get_checkpoint_and_cfg(model_size: str):
    if model_size == "tiny":
        checkpoint = os.path.join(APP_ROOT, "checkpoints/sam2.1_hiera_tiny.pt")
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    elif model_size == "small":
        checkpoint = os.path.join(APP_ROOT, "checkpoints/sam2.1_hiera_small.pt")
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    elif model_size == "large":
        checkpoint = os.path.join(APP_ROOT, "checkpoints/sam2.1_hiera_large.pt")
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    else:  # base_plus (default)
        checkpoint = os.path.join(APP_ROOT, "checkpoints/sam2.1_hiera_base_plus.pt")
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    return checkpoint, model_cfg


class ImageSegmentor:
    """
    Lightweight wrapper around SAM2 image predictor to segment a single image
    using interactive point prompts.
    """

    def __init__(self) -> None:
        checkpoint, model_cfg = _get_checkpoint_and_cfg(MODEL_SIZE)

        force_cpu_device = os.environ.get("SAM2_DEMO_FORCE_CPU_DEVICE", "0") == "1"
        if force_cpu_device:
            logger.info("forcing CPU device for SAM 2 demo (image)")
        
        # Detect if running in a forked process (e.g., gunicorn worker)
        # MPS (Metal) doesn't work correctly in forked processes on macOS
        # because the Metal compiler service crashes in forked children
        is_forked_process = False
        if torch.backends.mps.is_available() and not force_cpu_device:
            # Multiple ways to detect gunicorn/forked workers:
            if "gunicorn" in os.environ.get("SERVER_SOFTWARE", "").lower():
                is_forked_process = True
            try:
                import sys
                if any('gunicorn' in arg.lower() for arg in sys.argv):
                    is_forked_process = True
                main_module = sys.modules.get('__main__', None)
                if main_module:
                    main_spec = getattr(main_module, '__spec__', None)
                    if main_spec and main_spec.name and 'gunicorn' in main_spec.name.lower():
                        is_forked_process = True
                    main_file = getattr(main_module, '__file__', '') or ''
                    if 'gunicorn' in main_file.lower():
                        is_forked_process = True
            except Exception:
                pass
            
            if is_forked_process:
                logger.warning(
                    "Detected gunicorn worker. MPS doesn't work in forked processes. "
                    "Falling back to CPU."
                )
        
        if torch.cuda.is_available() and not force_cpu_device:
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and not force_cpu_device and not is_forked_process:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.info(f"using device for image segmentation: {device}")

        if device.type == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.device = device
        # Build base SAM2 model then wrap image predictor
        sam_model = build_sam2(
            config_file=model_cfg,
            ckpt_path=checkpoint,
            device=device,
            mode="eval",
        )
        sam_model.to(device)
        self.predictor = SAM2ImagePredictor(sam_model)
        self.lock = Lock()

    def autocast_context(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        elif self.device.type == "mps":
            # MPS supports float16 autocast (bfloat16 not fully supported on MPS)
            return torch.autocast("mps", dtype=torch.float16)
        return contextlib.nullcontext()

    def segment(self, image_b64: str, points: Tuple[Tuple[float, float], ...], labels: Tuple[int, ...], pad: int = 0):
        """
        Segment the image using point prompts (labels: 1=fg, 0=bg).
        Returns (png_base64, bbox) where bbox is (min_x, min_y, max_x, max_y).
        """
        raw = image_b64.split(",")[-1]
        image_bytes = base64.b64decode(raw)

        with self.autocast_context(), self.lock:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            np_img = np.array(pil_img)
            self.predictor.set_image(np_img)

            if len(points) == 0:
                raise ValueError("points must not be empty")

            point_coords = np.array(points, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int64)

            masks, _, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
            mask = masks[0]  # (H, W)

            # Compute tight bounding box from mask
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                bbox = (0, 0, np_img.shape[1] - 1, np_img.shape[0] - 1)
            else:
                min_x = int(xs.min())
                max_x = int(xs.max())
                min_y = int(ys.min())
                max_y = int(ys.max())
                if pad > 0:
                    min_x = max(0, min_x - pad)
                    min_y = max(0, min_y - pad)
                    max_x = min(np_img.shape[1] - 1, max_x + pad)
                    max_y = min(np_img.shape[0] - 1, max_y + pad)
                bbox = (min_x, min_y, max_x, max_y)

            alpha = (mask * 255).astype(np.uint8)
            rgba = np.dstack([np_img, alpha])
            out_img = Image.fromarray(rgba, mode="RGBA")
            buf = io.BytesIO()
            out_img.save(buf, format="PNG")
            png_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return png_b64, bbox

