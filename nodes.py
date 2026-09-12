"""The nodes. Classic ComfyUI node API (INPUT_TYPES / RETURN_TYPES / FUNCTION) so the pack
loads on every ComfyUI version; the VIDEO type is imported lazily so the IMAGE node still
works on builds that predate it.

Why the model is cached at module level: ComfyUI re-executes a node whenever an input
changes; reloading 111 MB of weights and rebuilding the CUDA context on every tweak of
`gamma` would cost seconds each time. One backend per (model, device) is kept for the
life of the process — it is 28M parameters, small next to any video model in the graph.
"""

from __future__ import annotations

import logging
import os
from fractions import Fraction

import numpy as np
import torch

from reshot import to_gray
from reshot.backends import get_backend, pick_device
from reshot.pipeline import QUALITY_INPUT_SIZE, model_input_resolution
from reshot.planning import processing_max_res
from reshot.targets import TARGETS, center_crop, fit_dimensions

log = logging.getLogger("ComfyUI-ReShot")

_BACKENDS: dict[tuple[str, str], object] = {}


def _device() -> str:
    """ComfyUI's idea of the compute device when running inside it; reshot's otherwise."""
    try:
        import comfy.model_management as mm

        d = mm.get_torch_device()
        return "cuda" if d.type == "cuda" else ("mps" if d.type == "mps" else "cpu")
    except Exception:
        return pick_device("auto")


def _backend(model: str = "small"):
    if os.environ.get("RESHOT_FAKE_BACKEND"):  # same test escape hatch as the reshot CLI
        return get_backend("fake")
    device = _device()
    key = (model, device)
    if key not in _BACKENDS:
        log.info("ReShot: loading Video Depth Anything %s on %s", model, device)
        _BACKENDS[key] = get_backend("vda", model=model, device=device)
    return _BACKENDS[key]


def _to_uint8_frames(images: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE `[T, H, W, 3]` float 0–1 → uint8 RGB numpy."""
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected an IMAGE batch [T, H, W, 3], got {tuple(images.shape)}")
    return (images.clamp(0, 1).mul(255).round().to(torch.uint8)).cpu().numpy()


def _resize_batch(frames: np.ndarray, width: int, height: int, interpolation: int) -> np.ndarray:
    import cv2

    if frames.shape[2] == width and frames.shape[1] == height:
        return frames
    out = np.empty((frames.shape[0], height, width) + frames.shape[3:], dtype=frames.dtype)
    for i in range(frames.shape[0]):
        out[i] = cv2.resize(frames[i], (width, height), interpolation=interpolation)
    return out


def depth_frames(
    frames: np.ndarray,
    *,
    quality: str = "fast",
    invert: bool = False,
    clip_percent: float = 0.0,
    gamma: float = 1.0,
    model: str = "small",
) -> np.ndarray:
    """uint8 RGB `[T, H, W, 3]` → uint8 grey depth `[T, H, W]` at the same size.

    Same policy as the CLI: infer at the model's working size (fast = 644×364 for 16:9,
    full = 924×518), normalise once over the whole clip, upsample the 8-bit result."""
    import cv2

    t, h, w = frames.shape[:3]
    input_size = QUALITY_INPUT_SIZE[quality]
    long_side = processing_max_res(w, h, input_size)
    scale = min(1.0, long_side / max(w, h))
    pw, ph = int(round(w * scale / 2) * 2), int(round(h * scale / 2) * 2)
    small = _resize_batch(frames, pw, ph, cv2.INTER_AREA)
    mw, mh = model_input_resolution(pw, ph, input_size)
    log.info("ReShot: %d frames, model sees %dx%d (%s)", t, mw, mh, quality)
    depths = _backend(model).infer(small, 24.0, input_size=input_size)
    gray = to_gray(depths, invert=invert, clip_percent=clip_percent, gamma=gamma)
    return _resize_batch(gray, w, h, cv2.INTER_LINEAR)


def _gray_to_image(gray: np.ndarray) -> torch.Tensor:
    """uint8 `[T, H, W]` → ComfyUI IMAGE `[T, H, W, 3]` float 0–1 (grey replicated)."""
    g = torch.from_numpy(np.ascontiguousarray(gray)).float().div_(255.0)
    return g.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()


def _crop_batch(gray: np.ndarray, multiple: int) -> np.ndarray:
    h, w = gray.shape[1:3]
    nh, nw = fit_dimensions(h, w, multiple)
    if (nh, nw) == (h, w):
        return gray
    return np.ascontiguousarray(center_crop(gray, nh, nw))  # works on the whole [T, H, W] batch


def _resample_by_timestamp(frames: np.ndarray, src_fps: float, target_fps: float) -> np.ndarray:
    """Pick frames by time, like reshot.io.read_video: 30 → 24 fps really is 24."""
    if target_fps <= 0 or target_fps >= src_fps:
        return frames
    keep, next_pick = [], 0.0
    for i in range(frames.shape[0]):
        t = i / src_fps
        if t + 1e-9 < next_pick:
            continue
        next_pick += 1.0 / target_fps
        keep.append(i)
    return frames[keep]


_QUALITY_TIP = "fast: model sees 644x364 (16:9), ~3 GB VRAM. full: 924x518, ~11 GB VRAM, 2.5x slower, sharper fine detail."
_INVERT_TIP = "Off: near is white (what depth ControlNets and Seedance/H3 expect). On: far is white."
_CLIP_TIP = "Percent trimmed from both ends before scaling to 0–255, so one hot pixel can't crush the contrast. 0 = none."
_GAMMA_TIP = ">1 darkens mid-tones (more separation near the camera). 1 = linear."


class ReShotDepthMap:
    """IMAGE batch → depth IMAGE batch. Drop it between any video loader and a depth ControlNet."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Video frames as an IMAGE batch [T, H, W, 3]."}),
                "quality": (["fast", "full"], {"default": "fast", "tooltip": _QUALITY_TIP}),
                "fit_to": (
                    ["none", "h3", "seedance", "wan"],
                    {"default": "none", "tooltip": "Center-crop to the model's frame-size multiple (h3: x32, seedance/wan: x16). fps is not changed here — use ReShot Depth Video for that."},
                ),
            },
            "optional": {
                "invert": ("BOOLEAN", {"default": False, "tooltip": _INVERT_TIP}),
                "clip_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": _CLIP_TIP}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05, "tooltip": _GAMMA_TIP}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "run"
    CATEGORY = "ReShot"
    DESCRIPTION = "Video frames → depth map frames (near = white), temporally consistent across the whole batch. Copy the shot, not the actors."

    def run(self, images, quality="fast", fit_to="none", invert=False, clip_percent=0.0, gamma=1.0):
        frames = _to_uint8_frames(images)
        gray = depth_frames(frames, quality=quality, invert=invert, clip_percent=clip_percent, gamma=gamma)
        if fit_to != "none":
            gray = _crop_batch(gray, TARGETS[fit_to].multiple)
        return (_gray_to_image(gray),)


class ReShotDepthVideo:
    """VIDEO → depth VIDEO with a generator preset applied (fps by timestamp + frame-size crop).

    Feed the output straight into a Seedance reference-video input, MiniMax H3's
    ReferenceToVideo (`ref_videos`) or a Save Video node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "From Load Video (or any node that outputs VIDEO)."}),
                "target": (
                    ["seedance", "h3", "wan", "none"],
                    {"default": "seedance", "tooltip": "Preset: seedance = 24 fps, x16; h3 = 24 fps, x32; wan = 16 fps, x16; none = keep fps, even dims."},
                ),
                "quality": (["fast", "full"], {"default": "fast", "tooltip": _QUALITY_TIP}),
            },
            "optional": {
                "invert": ("BOOLEAN", {"default": False, "tooltip": _INVERT_TIP}),
                "clip_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": _CLIP_TIP}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05, "tooltip": _GAMMA_TIP}),
                "max_side": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 2, "tooltip": "Cap the output's longer side (0 = source size). e.g. 320 for MiniMax H3 reference videos."}),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "FLOAT")
    RETURN_NAMES = ("depth_video", "depth_frames", "fps")
    FUNCTION = "run"
    CATEGORY = "ReShot"
    DESCRIPTION = "Reference video → depth-map video that meets the Seedance / MiniMax H3 / Wan reference-video rules. Copy the shot, not the actors."

    def run(self, video, target="seedance", quality="fast", invert=False, clip_percent=0.0, gamma=1.0, max_side=0):
        import cv2

        comp = video.get_components()
        src_fps = float(comp.frame_rate)
        frames = _to_uint8_frames(comp.images)
        preset = TARGETS[target]
        out_fps = src_fps if not preset.fps or preset.fps >= src_fps else float(preset.fps)
        frames = _resample_by_timestamp(frames, src_fps, out_fps)
        if preset.max_seconds and frames.shape[0] / out_fps > preset.max_seconds + 1e-6:
            log.warning("ReShot: clip is %.1fs; %s accepts <= %.0fs", frames.shape[0] / out_fps, target, preset.max_seconds)
        gray = depth_frames(frames, quality=quality, invert=invert, clip_percent=clip_percent, gamma=gamma)
        if max_side and max(gray.shape[1:3]) > max_side:
            s = max_side / max(gray.shape[1:3])
            gray = _resize_batch(gray, int(round(gray.shape[2] * s / 2) * 2), int(round(gray.shape[1] * s / 2) * 2), cv2.INTER_AREA)
        gray = _crop_batch(gray, preset.multiple)
        images = _gray_to_image(gray)
        return (_make_video(images, out_fps), images, out_fps)


def _make_video(images: torch.Tensor, fps: float):
    """Build a ComfyUI VIDEO from frames. The import paths moved between ComfyUI versions;
    try the current ones first, then the compatibility shims."""
    try:
        from comfy_api.latest import InputImpl, Types

        return InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=None, frame_rate=Fraction(fps)))
    except Exception:
        from comfy_api.input_impl import VideoFromComponents
        from comfy_api.util import VideoComponents

        return VideoFromComponents(VideoComponents(images=images, audio=None, frame_rate=Fraction(fps)))


NODE_CLASS_MAPPINGS = {
    "ReShotDepthMap": ReShotDepthMap,
    "ReShotDepthVideo": ReShotDepthVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ReShotDepthMap": "ReShot Depth Map",
    "ReShotDepthVideo": "ReShot Depth Video",
}
