# SPDX-License-Identifier: Apache-2.0
"""
Real-ESRGAN upscaling for SGLang diffusion pipelines.

SRVGGNetCompact model code is vendored and adapted from:
  - https://github.com/xinntao/Real-ESRGAN  (BSD-3-Clause License)
  Copyright (c) 2021 Xintao Wang

The RealESRGANUpscaler wrapper and integration code are original work.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_DEFAULT_HF_REPO = "leonelhs/realesrgan"
_DEFAULT_WEIGHT_FILENAME = "realesr-animevideov3.pth"

_MODEL_CACHE: dict[str, "SRVGGNetCompact"] = {}


# ---------------------------------------------------------------------------
# Vendored SRVGGNetCompact (Real-ESRGAN animevideov3 backbone)
# ---------------------------------------------------------------------------


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style super-resolution network (Real-ESRGAN).

    Performs upsampling in the last layer via PixelShuffle and learns the
    residual on top of nearest-neighbor upsampled input.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: str = "prelu",
    ):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.upscale = upscale

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(self._make_activation(act_type, num_feat))

        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._make_activation(act_type, num_feat))

        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _make_activation(act_type: str, num_feat: int) -> nn.Module:
        if act_type == "relu":
            return nn.ReLU(inplace=True)
        elif act_type == "prelu":
            return nn.PReLU(num_parameters=num_feat)
        elif act_type == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        raise ValueError(f"Unknown activation: {act_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


# ---------------------------------------------------------------------------
# Tiled inference helper
# ---------------------------------------------------------------------------


def _tile_forward(
    model: SRVGGNetCompact,
    img: torch.Tensor,
    tile_size: int,
    tile_pad: int = 10,
) -> torch.Tensor:
    """Run model on overlapping tiles and stitch the result.

    Args:
        model: The upscaler network.
        img: Input tensor of shape [1, C, H, W].
        tile_size: Tile edge length in pixels (input space).
        tile_pad: Overlap padding on each side to avoid seam artifacts.

    Returns:
        Upscaled tensor of shape [1, C, H*scale, W*scale].
    """
    scale = model.upscale
    _, _, h, w = img.shape
    output = img.new_zeros(1, img.shape[1], h * scale, w * scale)
    tiles_x = max(1, (w + tile_size - 1) // tile_size)
    tiles_y = max(1, (h + tile_size - 1) // tile_size)

    for yi in range(tiles_y):
        for xi in range(tiles_x):
            x_start = xi * tile_size
            y_start = yi * tile_size
            x_end = min(x_start + tile_size, w)
            y_end = min(y_start + tile_size, h)

            # Expand tile with padding
            x_start_pad = max(x_start - tile_pad, 0)
            y_start_pad = max(y_start - tile_pad, 0)
            x_end_pad = min(x_end + tile_pad, w)
            y_end_pad = min(y_end + tile_pad, h)

            tile_in = img[:, :, y_start_pad:y_end_pad, x_start_pad:x_end_pad]
            tile_out = model(tile_in)

            # Crop away the padding in output space
            out_y_start = (y_start - y_start_pad) * scale
            out_x_start = (x_start - x_start_pad) * scale
            out_y_end = out_y_start + (y_end - y_start) * scale
            out_x_end = out_x_start + (x_end - x_start) * scale

            output[
                :,
                :,
                y_start * scale : y_end * scale,
                x_start * scale : x_end * scale,
            ] = tile_out[:, :, out_y_start:out_y_end, out_x_start:out_x_end]

    return output


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _resolve_weight_path(model_path: Optional[str]) -> str:
    """Resolve a local path or HF repo ID to a local weight file path.

    If *model_path* is a local file, return it directly.  If it is a local
    directory, look for the default weight filename inside it.  Otherwise
    treat it as a HuggingFace repo ID and download the weight file.
    When *model_path* is None, download from the default HF repo.
    """
    if model_path is None:
        repo_id = _DEFAULT_HF_REPO
        filename = _DEFAULT_WEIGHT_FILENAME
    elif os.path.isfile(model_path):
        return model_path
    elif os.path.isdir(model_path):
        candidate = os.path.join(model_path, _DEFAULT_WEIGHT_FILENAME)
        if os.path.isfile(candidate):
            return candidate
        pth_files = [f for f in os.listdir(model_path) if f.endswith(".pth")]
        if len(pth_files) == 1:
            return os.path.join(model_path, pth_files[0])
        raise FileNotFoundError(
            f"Cannot find Real-ESRGAN weights in {model_path}. "
            f"Expected '{_DEFAULT_WEIGHT_FILENAME}' or a single .pth file."
        )
    else:
        # Treat as HF repo ID.  If it contains '/' with a filename part
        # (e.g. "user/repo/file.pth") split it; otherwise use defaults.
        repo_id = model_path
        filename = _DEFAULT_WEIGHT_FILENAME

    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    logger.info("Downloaded Real-ESRGAN weights to %s", local_path)
    return local_path


def _load_model(weight_path: str, device: torch.device) -> SRVGGNetCompact:
    """Instantiate SRVGGNetCompact and load pretrained weights."""
    model = SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=16,
        upscale=4,
        act_type="prelu",
    )
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    # Handle state dicts wrapped in a top-level key (e.g. "params_ema")
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]
    model.load_state_dict(state, strict=True)
    model.eval()
    model = model.to(device)
    logger.info("Loaded Real-ESRGAN weights from %s", weight_path)
    return model


# ---------------------------------------------------------------------------
# RealESRGANUpscaler public class
# ---------------------------------------------------------------------------


class RealESRGANUpscaler:
    """Lazy-loaded Real-ESRGAN upscaler.

    Weights are loaded on first call to `upscale()` and cached globally
    per weight path to avoid reloading across requests.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path

    def _ensure_model_loaded(self) -> SRVGGNetCompact:
        weight_path = _resolve_weight_path(self._model_path)

        if weight_path in _MODEL_CACHE:
            return _MODEL_CACHE[weight_path]

        device = current_platform.get_local_torch_device()
        model = _load_model(weight_path, device)
        _MODEL_CACHE[weight_path] = model
        logger.info("Real-ESRGAN model loaded on device: %s", device)
        return model

    def upscale(
        self,
        frames: list[np.ndarray],
        outscale: int = 4,
        tile_size: int = 0,
    ) -> list[np.ndarray]:
        """Upscale a list of uint8 HWC numpy frames.

        Args:
            frames: List of uint8 numpy arrays with shape [H, W, 3].
            outscale: Desired output scale factor (2 or 4).
            tile_size: Tile size for tiled inference (0 = no tiling).

        Returns:
            List of upscaled uint8 HWC numpy frames.
        """
        if not frames:
            return frames

        model = self._ensure_model_loaded()
        device = next(model.parameters()).device
        net_scale = model.upscale  # native model scale (4)

        results: list[np.ndarray] = []
        for frame in frames:
            img = (
                torch.from_numpy(frame)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .div_(255.0)
                .to(device)
            )

            with torch.no_grad():
                if tile_size > 0:
                    output = _tile_forward(model, img, tile_size)
                else:
                    output = model(img)

            # If desired scale differs from native scale, resize
            if outscale != net_scale:
                h, w = frame.shape[:2]
                target_h, target_w = h * outscale, w * outscale
                output = F.interpolate(
                    output,
                    size=(target_h, target_w),
                    mode="bicubic",
                    align_corners=False,
                )

            arr = (
                output.squeeze(0)
                .permute(1, 2, 0)
                .clamp(0.0, 1.0)
                .mul_(255.0)
                .byte()
                .cpu()
                .numpy()
            )
            results.append(arr)

        return results


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def upscale_frames(
    frames: list[np.ndarray],
    outscale: int = 4,
    tile_size: int = 0,
    model_path: Optional[str] = None,
) -> list[np.ndarray]:
    """Convenience wrapper around RealESRGANUpscaler.

    Args:
        frames:     List of uint8 HWC numpy frames.
        outscale:   Output scale factor (default 4).
        tile_size:  Tile size for inference (0 = no tiling).
        model_path: Local path or HuggingFace repo ID for weights.
                    None uses the default ``leonelhs/realesrgan``.

    Returns:
        List of upscaled uint8 HWC numpy frames.
    """
    upscaler = RealESRGANUpscaler(model_path=model_path)
    return upscaler.upscale(frames, outscale=outscale, tile_size=tile_size)
