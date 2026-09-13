"""ComfyUI-ReShot — ReShot depth maps, skeletons and lines as ComfyUI nodes.

Six nodes, all backed by the `reshot` package (https://github.com/maosika-ai/reshot):

* **ReShot Depth Map / Depth Video**  depth (near = white), Video Depth Anything Small.
* **ReShot Pose Map / Pose Video**    OpenPose-style skeletons, DWPose, tracked and smoothed; also
                                      returns the keypoints as JSON.
* **ReShot Canny Map / Canny Video**  white edge lines, OpenCV, no model.

"Map" nodes are IMAGE batch → IMAGE batch; "Video" nodes are VIDEO → VIDEO with the Seedance /
MiniMax H3 / Wan presets (fps and frame-size rules) applied. Models load once per process.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
