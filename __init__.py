"""ComfyUI-ReShot — ReShot depth maps as ComfyUI nodes.

Two nodes, both backed by the `reshot` package (https://github.com/maosika-ai/reshot):

* **ReShot Depth Map**   IMAGE batch → IMAGE batch of depth (near = white).
* **ReShot Depth Video** VIDEO → VIDEO of depth, with the Seedance / MiniMax H3 / Wan presets
  (fps and frame-size rules) applied, ready to plug into a reference-video or ControlNet input.

The model (Video Depth Anything Small, Apache-2.0) is loaded once per process and reused.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
