"""The nodes. Classic ComfyUI node API (INPUT_TYPES / RETURN_TYPES / FUNCTION) so the pack
loads on every ComfyUI version; the VIDEO type is imported lazily so the IMAGE nodes still
work on builds that predate it.

Three control types, each as an IMAGE→IMAGE "Map" node and a VIDEO→VIDEO "Video" node
that additionally applies a generator preset (fps by timestamp + frame-size crop):

    Depth   Video Depth Anything → grey, near = white          (reshot >= 0.3)
    Pose    DWPose → OpenPose-style coloured skeletons on black (reshot >= 0.5, needs onnxruntime)
    Canny   OpenCV edges → white lines on black                 (reshot >= 0.5, no model)

Why the models are cached at module level: ComfyUI re-executes a node whenever an input
changes; reloading weights and rebuilding the CUDA context on every tweak of `gamma`
would cost seconds each time. One backend per (kind, model, device) is kept for the life
of the process — 28M parameters for depth, two ONNX files for pose, small next to any
video model in the graph.
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
    key = ("vda", model, device)
    if key not in _BACKENDS:
        log.info("ReShot: loading Video Depth Anything %s on %s", model, device)
        _BACKENDS[key] = get_backend("vda", model=model, device=device)
    return _BACKENDS[key]


def _pose_backend(detect_every: int = 3):
    if os.environ.get("RESHOT_FAKE_BACKEND"):
        return get_backend("fake-pose")
    device = _device()
    key = ("dwpose", detect_every, device)
    if key not in _BACKENDS:
        log.info("ReShot: loading DWPose on %s", device)
        _BACKENDS[key] = get_backend("dwpose", device=device, detect_every=detect_every)
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


def pose_frames(
    frames: np.ndarray,
    *,
    hands: bool = True,
    face: bool = False,
    smooth: bool = True,
    detect_every: int = 3,
) -> tuple[np.ndarray, str]:
    """uint8 RGB `[T, H, W, 3]` → (uint8 RGB skeleton frames `[T, H, W, 3]`, keypoints JSON).

    Same chain as the CLI's `--control pose`: DWPose per frame (detector every N frames,
    skeleton-tracked boxes in between, cuts re-detect), then track → visibility
    hysteresis → One-Euro smoothing, then the OpenPose render at the source size."""
    import json

    from reshot.pose.skeleton import render_frame
    from reshot.pose.tracking import stabilise

    t, h, w = frames.shape[:3]
    clip = _pose_backend(detect_every).infer(frames, 24.0)
    if smooth:
        stabilise(clip)
    people = [k.shape[0] for k in clip.keypoints]
    log.info("ReShot: pose on %d frames, people per frame %d–%d", t, min(people), max(people))
    out = np.empty((t, h, w, 3), dtype=np.uint8)
    for i in range(t):
        out[i] = render_frame(clip.keypoints[i], clip.scores[i], h, w, hands=hands, face=face)
    return out, json.dumps(clip.to_json_dict(), separators=(",", ":"))


def canny_frames(frames: np.ndarray, *, low: int = 100, high: int = 200) -> np.ndarray:
    """uint8 RGB `[T, H, W, 3]` → uint8 edge frames `[T, H, W]` (white lines on black)."""
    from reshot.edges import canny_frame

    t, h, w = frames.shape[:3]
    out = np.empty((t, h, w), dtype=np.uint8)
    for i in range(t):
        out[i] = canny_frame(frames[i], h, w, low=low, high=high)
    return out


def _rgb_to_image(rgb: np.ndarray) -> torch.Tensor:
    """uint8 `[T, H, W, 3]` → ComfyUI IMAGE float 0–1."""
    return torch.from_numpy(np.ascontiguousarray(rgb)).float().div_(255.0)


def _gray_to_image(gray: np.ndarray) -> torch.Tensor:
    """uint8 `[T, H, W]` → ComfyUI IMAGE `[T, H, W, 3]` float 0–1 (grey replicated)."""
    g = torch.from_numpy(np.ascontiguousarray(gray)).float().div_(255.0)
    return g.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()


def _crop_batch(arr: np.ndarray, multiple: int) -> np.ndarray:
    h, w = arr.shape[1:3]
    nh, nw = fit_dimensions(h, w, multiple)
    if (nh, nw) == (h, w):
        return arr
    return np.ascontiguousarray(center_crop(arr, nh, nw))  # works on the whole [T, H, W(, 3)] batch


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


def _video_in(video, target: str):
    """VIDEO → (uint8 frames resampled to the preset fps, out_fps, preset)."""
    comp = video.get_components()
    src_fps = float(comp.frame_rate)
    frames = _to_uint8_frames(comp.images)
    preset = TARGETS[target]
    out_fps = src_fps if not preset.fps or preset.fps >= src_fps else float(preset.fps)
    frames = _resample_by_timestamp(frames, src_fps, out_fps)
    if preset.max_seconds and frames.shape[0] / out_fps > preset.max_seconds + 1e-6:
        log.warning("ReShot: clip is %.1fs; %s accepts <= %.0fs", frames.shape[0] / out_fps, target, preset.max_seconds)
    return frames, out_fps, preset


def _fit_out(arr: np.ndarray, preset, max_side: int) -> np.ndarray:
    """Cap the longer side, then centre-crop to the preset's multiple. Works on [T, H, W] and [T, H, W, 3]."""
    import cv2

    if max_side and max(arr.shape[1:3]) > max_side:
        sc = max_side / max(arr.shape[1:3])
        arr = _resize_batch(arr, int(round(arr.shape[2] * sc / 2) * 2), int(round(arr.shape[1] * sc / 2) * 2), cv2.INTER_AREA)
    return _crop_batch(arr, preset.multiple)


_TARGET_TIP = "Preset: seedance = 24 fps, x16; h3 = 24 fps, x32; wan = 16 fps, x16; none = keep fps, even dims."
_MAX_SIDE_TIP = "Cap the output's longer side (0 = source size). e.g. 320 for MiniMax H3 reference videos."
_HANDS_TIP = "Draw the 21-point hands (grips and gestures)."
_FACE_TIP = "Draw the 68 face points. Off by default: the face shape is what ReShot throws away."
_SMOOTH_TIP = "Track people across frames, hold joints that dip briefly, One-Euro smooth. Off = raw per-frame output."
_DETECT_TIP = "Run the person detector every N frames and follow the skeletons in between (cuts always re-detect). 1 = every frame, ~2.5x slower on CPU."
_CANNY_LOW_TIP = "Canny low threshold: lower = more lines."
_CANNY_HIGH_TIP = "Canny high threshold: higher = only strong edges."

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
        frames, out_fps, preset = _video_in(video, target)
        gray = depth_frames(frames, quality=quality, invert=invert, clip_percent=clip_percent, gamma=gamma)
        gray = _fit_out(gray, preset, max_side)
        images = _gray_to_image(gray)
        return (_make_video(images, out_fps), images, out_fps)


class ReShotPoseMap:
    """IMAGE batch → OpenPose-style skeleton IMAGE batch (DWPose). Drop it before a pose ControlNet."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Video frames as an IMAGE batch [T, H, W, 3]."}),
                "fit_to": (
                    ["none", "h3", "seedance", "wan"],
                    {"default": "none", "tooltip": "Center-crop to the model's frame-size multiple (h3: x32, seedance/wan: x16). fps is not changed here — use ReShot Pose Video for that."},
                ),
            },
            "optional": {
                "hands": ("BOOLEAN", {"default": True, "tooltip": _HANDS_TIP}),
                "face": ("BOOLEAN", {"default": False, "tooltip": _FACE_TIP}),
                "smooth": ("BOOLEAN", {"default": True, "tooltip": _SMOOTH_TIP}),
                "detect_every": ("INT", {"default": 3, "min": 1, "max": 30, "tooltip": _DETECT_TIP}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("pose", "keypoints_json")
    FUNCTION = "run"
    CATEGORY = "ReShot"
    DESCRIPTION = "Video frames → OpenPose-style skeleton frames (coloured stick figures on black, hands included, face not), tracked and smoothed over time. For MiniMax H3 Fun ControlNet pose, Wan VACE and any pose ControlNet."

    def run(self, images, fit_to="none", hands=True, face=False, smooth=True, detect_every=3):
        frames = _to_uint8_frames(images)
        rgb, kp = pose_frames(frames, hands=hands, face=face, smooth=smooth, detect_every=detect_every)
        if fit_to != "none":
            rgb = _crop_batch(rgb, TARGETS[fit_to].multiple)
        return (_rgb_to_image(rgb), kp)


class ReShotPoseVideo:
    """VIDEO → skeleton VIDEO with a generator preset applied."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "From Load Video (or any node that outputs VIDEO)."}),
                "target": (["seedance", "h3", "wan", "none"], {"default": "h3", "tooltip": _TARGET_TIP}),
            },
            "optional": {
                "hands": ("BOOLEAN", {"default": True, "tooltip": _HANDS_TIP}),
                "face": ("BOOLEAN", {"default": False, "tooltip": _FACE_TIP}),
                "smooth": ("BOOLEAN", {"default": True, "tooltip": _SMOOTH_TIP}),
                "detect_every": ("INT", {"default": 3, "min": 1, "max": 30, "tooltip": _DETECT_TIP}),
                "max_side": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 2, "tooltip": _MAX_SIDE_TIP}),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("pose_video", "pose_frames", "fps", "keypoints_json")
    FUNCTION = "run"
    CATEGORY = "ReShot"
    DESCRIPTION = "Reference video → OpenPose-style skeleton video that meets the Seedance / MiniMax H3 / Wan rules. Copy the moves, not the actors."

    def run(self, video, target="h3", hands=True, face=False, smooth=True, detect_every=3, max_side=0):
        frames, out_fps, preset = _video_in(video, target)
        rgb, kp = pose_frames(frames, hands=hands, face=face, smooth=smooth, detect_every=detect_every)
        rgb = _fit_out(rgb, preset, max_side)
        images = _rgb_to_image(rgb)
        return (_make_video(images, out_fps), images, out_fps, kp)


class ReShotCannyMap:
    """IMAGE batch → white-line edge IMAGE batch. No model; instant."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Video frames as an IMAGE batch [T, H, W, 3]."}),
                "fit_to": (["none", "h3", "seedance", "wan"], {"default": "none", "tooltip": "Center-crop to the model's frame-size multiple."}),
            },
            "optional": {
                "low": ("INT", {"default": 100, "min": 0, "max": 1000, "step": 10, "tooltip": _CANNY_LOW_TIP}),
                "high": ("INT", {"default": 200, "min": 0, "max": 1000, "step": 10, "tooltip": _CANNY_HIGH_TIP}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("canny",)
    FUNCTION = "run"
    CATEGORY = "ReShot"
    DESCRIPTION = "Video frames → canny line frames (white on black). Carries the outline of clothes and faces — depth or pose if you want those gone."

    def run(self, images, fit_to="none", low=100, high=200):
        gray = canny_frames(_to_uint8_frames(images), low=low, high=max(low, high))
        if fit_to != "none":
            gray = _crop_batch(gray, TARGETS[fit_to].multiple)
        return (_gray_to_image(gray),)


class ReShotCannyVideo:
    """VIDEO → canny line VIDEO with a generator preset applied."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "From Load Video (or any node that outputs VIDEO)."}),
                "target": (["seedance", "h3", "wan", "none"], {"default": "h3", "tooltip": _TARGET_TIP}),
            },
            "optional": {
                "low": ("INT", {"default": 100, "min": 0, "max": 1000, "step": 10, "tooltip": _CANNY_LOW_TIP}),
                "high": ("INT", {"default": 200, "min": 0, "max": 1000, "step": 10, "tooltip": _CANNY_HIGH_TIP}),
                "max_side": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 2, "tooltip": _MAX_SIDE_TIP}),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "FLOAT")
    RETURN_NAMES = ("canny_video", "canny_frames", "fps")
    FUNCTION = "run"
    CATEGORY = "ReShot"
    DESCRIPTION = "Reference video → canny line video that meets the Seedance / MiniMax H3 / Wan rules."

    def run(self, video, target="h3", low=100, high=200, max_side=0):
        frames, out_fps, preset = _video_in(video, target)
        gray = _fit_out(canny_frames(frames, low=low, high=max(low, high)), preset, max_side)
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
    "ReShotPoseMap": ReShotPoseMap,
    "ReShotPoseVideo": ReShotPoseVideo,
    "ReShotCannyMap": ReShotCannyMap,
    "ReShotCannyVideo": ReShotCannyVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ReShotDepthMap": "ReShot Depth Map",
    "ReShotDepthVideo": "ReShot Depth Video",
    "ReShotPoseMap": "ReShot Pose Map",
    "ReShotPoseVideo": "ReShot Pose Video",
    "ReShotCannyMap": "ReShot Canny Map",
    "ReShotCannyVideo": "ReShot Canny Video",
}
