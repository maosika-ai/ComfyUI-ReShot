"""Node logic without ComfyUI: the fake reshot backend stands in for the model, a tiny
stand-in object for VIDEO. Run: RESHOT_FAKE_BACKEND=1 python -m pytest -q tests"""

import os
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["RESHOT_FAKE_BACKEND"] = "1"

import nodes  # noqa: E402


def _images(t=12, h=120, w=200):
    return torch.rand(t, h, w, 3)


def test_depth_map_node_keeps_size_and_is_grey():
    out, = nodes.ReShotDepthMap().run(_images(), quality="fast")
    assert out.shape == (12, 120, 200, 3) and out.dtype == torch.float32
    assert torch.equal(out[..., 0], out[..., 1]) and torch.equal(out[..., 1], out[..., 2])
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_depth_map_node_fit_to_h3_crops_to_multiple_of_32():
    out, = nodes.ReShotDepthMap().run(_images(h=120, w=200), fit_to="h3")
    assert out.shape[1:3] == (96, 192)


def test_depth_video_node_applies_preset_fps_and_crop(monkeypatch):
    captured = {}

    def fake_make_video(images, fps):
        captured["images"], captured["fps"] = images, fps
        return "VIDEO"

    monkeypatch.setattr(nodes, "_make_video", fake_make_video)
    video = SimpleNamespace(get_components=lambda: SimpleNamespace(images=_images(t=30), frame_rate=Fraction(30), audio=None))
    v, frames, fps = nodes.ReShotDepthVideo().run(video, target="h3", quality="fast")
    assert v == "VIDEO" and fps == 24.0
    assert 23 <= frames.shape[0] <= 25  # 30 → 24 fps by timestamp
    assert frames.shape[1:3] == (96, 192)  # ×32
    assert captured["fps"] == 24.0 and captured["images"].shape == frames.shape


def test_depth_video_node_max_side(monkeypatch):
    monkeypatch.setattr(nodes, "_make_video", lambda images, fps: "VIDEO")
    video = SimpleNamespace(get_components=lambda: SimpleNamespace(images=_images(t=4, h=496, w=864), frame_rate=Fraction(24), audio=None))
    _, frames, _ = nodes.ReShotDepthVideo().run(video, target="h3", max_side=320)
    assert frames.shape[1:3] == (160, 320)  # 320×184 → ×32 crop → 320×160


def test_mappings_are_exported():
    from importlib import import_module

    pkg = import_module("nodes")
    assert set(pkg.NODE_CLASS_MAPPINGS) == {"ReShotDepthMap", "ReShotDepthVideo"}
    with pytest.raises(ValueError):
        nodes._to_uint8_frames(torch.rand(3, 4, 5))
