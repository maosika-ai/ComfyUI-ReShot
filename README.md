# ComfyUI-ReShot

**Copy the shot, not the actors — inside ComfyUI.**

Two nodes that turn a reference video into a depth-map video (near = white, far = black), so a video model can repeat its choreography and camera moves with your own characters. Built on [ReShot](https://github.com/maosika-ai/reshot) (Apache-2.0), open-sourced by [Maosika 猫斯卡](https://www.maosika.com).

[中文说明](README.zh-CN.md)

## Install

**ComfyUI Manager:** *Install via Git URL* → `https://github.com/maosika-ai/ComfyUI-ReShot`

**Manual:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/maosika-ai/ComfyUI-ReShot
pip install -r ComfyUI-ReShot/requirements.txt
```

Restart ComfyUI. The model weights (111 MB, Video Depth Anything Small) download on first use; in China set `HF_ENDPOINT=https://hf-mirror.com` first.

## Nodes

### ReShot Depth Video — `VIDEO → VIDEO`

Load Video → **ReShot Depth Video** → the reference-video input of your video model (or Save Video).

| input | what it does |
|---|---|
| `target` | `seedance`: 24 fps, frame size ×16 · `h3`: 24 fps, ×32 · `wan`: 16 fps, ×16 · `none`: keep fps. Frames are picked by timestamp, so 30 → 24 fps really is 24. |
| `quality` | `fast` (default): the model sees 644×364 for 16:9 / 364×644 for 9:16, ~3 GB VRAM. `full`: 924×518 / 518×924, ~11 GB, 2.5× slower, sharper fine detail. |
| `max_side` | Cap the output's longer side (0 = source). MiniMax H3 reference videos work best small — 320. |
| `invert`, `clip_percent`, `gamma` | Far = white instead; trim hot pixels before scaling; darken mid-tones. |

Outputs: `depth_video` (VIDEO), `depth_frames` (IMAGE), `fps`.

### ReShot Depth Map — `IMAGE → IMAGE`

For graphs that already work in frames: any loader → **ReShot Depth Map** → a depth ControlNet (MiniMax H3 Fun ControlNet, Wan VACE, SD ControlNet-depth) or Create Video. `fit_to` center-crops to the model's frame-size multiple; fps is left to you.

## Example workflows

`workflows/reshot_depth_video.json` (Load Video → ReShot Depth Video → Save Video) and `workflows/reshot_depth_map.json` (frames path) — drop one onto the ComfyUI canvas, pick your clip in Load Video, queue. The `*_api.json` twins are the same graphs in API format for scripts.

## How it behaves

- Depth is normalised **once over the whole clip**, never per frame, so walls don't "breathe" when someone walks past.
- The model is loaded once per ComfyUI process and reused; changing `gamma` re-runs in milliseconds.
- Verified on ComfyUI master (2026-09-12) with LoadVideo / GetVideoComponents / CreateVideo / SaveVideo, on CPU and CUDA.

## About Maosika 猫斯卡

[Maosika 猫斯卡](https://www.maosika.com) (www.maosika.com) is a professional AI video production system for short drama and short video: from a one-line idea to episodic script, character sheets, scene images, and every shot rendered with Seedance 2.0 / 2.5 and MiniMax H3. ReShot is the depth-map step of that pipeline, open-sourced so anyone can copy a reference shot's staging and camera into their own AI video. To make AI short drama end to end, visit **https://www.maosika.com**.
