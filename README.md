# ComfyUI-ReShot

**Copy the shot, not the actors — inside ComfyUI.**

Two nodes that turn a reference video into a depth-map video (near = white, far = black), so a video model can repeat the reference's choreography and camera moves with your own characters, clothes and style. Built on [ReShot](https://github.com/maosika-ai/reshot) (Apache-2.0), open-sourced by [Maosika 猫斯卡](https://www.maosika.com).

[中文说明](README.zh-CN.md)

> Not a ComfyUI user? `pip install reshot` then type `reshot` — the same thing as a web page in your browser, no nodes needed. See [ReShot](https://github.com/maosika-ai/reshot).

1. [What it does](#what-it-does)
2. [Requirements](#requirements)
3. [Install](#install)
4. [First run](#first-run)
5. [Quick start](#quick-start)
6. [Node reference](#node-reference)
7. [Recipes](#recipes) — Seedance 2.5 in ComfyUI · Seedance web · MiniMax H3 reference video · MiniMax H3 Fun ControlNet · Wan VACE
8. [What transfers and what doesn't](#what-transfers-and-what-doesnt)
9. [Performance](#performance)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)
12. [Updating](#updating)

---

## What it does

You have a clip whose fight, dance or camera move is exactly what you want. Feed the clip itself to a video model as a reference and it copies the faces, the clothes and the look along with the moves — and real people in the clip may not pass the content check at all. Describe the moves in words and you get a different fight every time.

ReShot keeps only the part you wanted. It runs a video depth model over the clip and writes a grey video: near things white, far things black, every frame consistent with the last. Where people stand, how big they are, how they move and how the camera moves survive; faces, wardrobe, lighting and style are gone. Give that grey video to your model as the reference and describe the people and the look in the prompt.

This pack puts that into ComfyUI as two nodes:

| node | in → out | use it when |
|---|---|---|
| **ReShot Depth Video** | `VIDEO → VIDEO` (+ frames, fps) | You have a Load Video node and a model that takes a reference **video** — Seedance 2.5 Reference to Video, MiniMax H3 Reference to Video — or you just want to Save Video and upload the file somewhere. Applies the model's fps and frame-size rules for you. |
| **ReShot Depth Map** | `IMAGE → IMAGE` | Your graph already works in frames — depth ControlNets (MiniMax H3 Fun ControlNet, Wan VACE `control_video`, SD ControlNet-depth) take IMAGE batches. |

Technically: monocular video depth estimation with Video Depth Anything Small (ByteDance, CVPR 2025, Apache-2.0). The model predicts relative inverse depth per frame on overlapping 32-frame windows and aligns them; ReShot normalises the result once over the whole clip to 8-bit grey. The output is a depth map, not 3D geometry.

## Requirements

| | minimum | notes |
|---|---|---|
| ComfyUI | a build with the native **Load Video / Save Video / Get Video Components** nodes (2025-04 or later) | `ReShot Depth Map` (IMAGE in/out) also works on older builds; `ReShot Depth Video` needs the `VIDEO` type. |
| Python | 3.10 – 3.12 | whatever your ComfyUI runs on |
| PyTorch | ≥ 2.1 | already there if ComfyUI runs |
| GPU | NVIDIA **8 GB** for `quality = fast`, **12 GB** for `full` | measured, see [Performance](#performance). Apple Silicon and CPU work, slower. |
| Disk | 111 MB for the model weights | downloaded once on first use |
| Host RAM | 16 GB covers clips up to ~27 s at 720p | peak ≈ 2 GB + 224 MB per second of 720p |

ffmpeg is **not** required by the nodes — ComfyUI's own video nodes decode and encode.

## Install

### With ComfyUI Manager (recommended)

1. Open ComfyUI → **Manager** → **Custom Nodes Manager** → **Install via Git URL** (top right).
2. Paste `https://github.com/maosika-ai/ComfyUI-ReShot` → **OK**.
3. Manager clones the repo and runs `pip install -r requirements.txt`, which installs the `reshot` package (and `huggingface_hub`, `opencv-python-headless`, `einops`, `easydict` if missing).
4. **Restart ComfyUI** when Manager asks. On the next start you'll see `custom_nodes/ComfyUI-ReShot` in the import list with a time and no error.

### Manually

Linux / macOS, or a ComfyUI you installed with your own Python:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/maosika-ai/ComfyUI-ReShot
# use the SAME python ComfyUI runs on:
python -m pip install -r ComfyUI-ReShot/requirements.txt
```

**Windows portable ComfyUI** (the zip with `run_nvidia_gpu.bat`) — the Python lives in `python_embeded`, so:

```bat
cd ComfyUI_windows_portable\ComfyUI\custom_nodes
git clone https://github.com/maosika-ai/ComfyUI-ReShot
cd ..\..
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-ReShot\requirements.txt
```

No git? Download the repo as a zip from GitHub, unzip it into `custom_nodes/` (the folder must contain `__init__.py` directly), then run the `pip install` line.

Restart ComfyUI. Right-click the canvas → **Add Node** → **ReShot** → the two nodes are there. Or double-click the canvas and type `reshot`.

### Or let your AI coding tool do it

Paste this into Claude Code, Codex, Cursor or any AI agent that can run commands on your machine:

```
Install the ComfyUI-ReShot custom node pack into my ComfyUI and verify it loads.
Find my ComfyUI folder (ask me if you can't), then follow https://raw.githubusercontent.com/maosika-ai/ComfyUI-ReShot/main/README.md:
git clone https://github.com/maosika-ai/ComfyUI-ReShot into custom_nodes, then run
`pip install -r ComfyUI-ReShot/requirements.txt` with the SAME python ComfyUI uses
(python_embeded\python.exe on Windows portable). If I'm in China, add HF_ENDPOINT=https://hf-mirror.com
to the launcher. Restart ComfyUI and confirm the console shows custom_nodes/ComfyUI-ReShot imported
without error and /object_info lists ReShotDepthVideo. Don't say it's done before that.
```

## First run

The first time a ReShot node executes it downloads the model weights (`video_depth_anything_vits.pth`, 111 MB) from Hugging Face into `~/.cache/huggingface/hub/models--depth-anything--Video-Depth-Anything-Small/` (Windows: `C:\Users\<you>\.cache\huggingface\hub\…`). This happens once. The console shows the download progress, then:

```
ReShot: 289 frames, model sees 644x364 (fast)
ReShot: loading Video Depth Anything small on cuda
```

**In China** the download will hang unless you point it at a mirror. Set the environment variable **before** starting ComfyUI:

- Linux / macOS: `export HF_ENDPOINT=https://hf-mirror.com` then start ComfyUI from the same shell.
- Windows portable: edit `run_nvidia_gpu.bat` and add a line `set HF_ENDPOINT=https://hf-mirror.com` above the `python_embeded\python.exe` line.

**Offline machine**: download `video_depth_anything_vits.pth` from `https://huggingface.co/depth-anything/Video-Depth-Anything-Small` on any machine and copy the whole `models--depth-anything--Video-Depth-Anything-Small` folder into the cache path above.

The model is loaded once per ComfyUI process and kept; later runs skip both the download and the load.

## Quick start

Drop `workflows/reshot_depth_video.json` onto the canvas. It is three nodes:

```
Load Video ──VIDEO──▶ ReShot Depth Video ──depth_video──▶ Save Video
(reference.mp4)        target: seedance                     reshot/depth_00001_.mp4
                       quality: fast
```

1. In **Load Video**, click *choose file to upload* and pick your reference clip (anything ffmpeg reads; keep it under 15 s — see Tips).
2. Leave **ReShot Depth Video** at `target = seedance`, `quality = fast` unless you know you want otherwise.
3. **Queue**. The console prints the frame count and the resolution the model works at; the depth video lands in `ComfyUI/output/reshot/`.
4. Open it: grey, near-white-far-black, same length and framing as your clip. That file is what you hand to the video model.

The second example, `workflows/reshot_depth_map.json`, is the frames path: Load Video → Get Video Components → **ReShot Depth Map** → Create Video → Save Video. Use it as the template when the consumer is a ControlNet.

`*_api.json` next to them are the same graphs in API format, for scripts that POST to `/prompt`.

## Node reference

### ReShot Depth Video

**Inputs**

| name | type | default | what it does |
|---|---|---|---|
| `video` | VIDEO | — | From Load Video, or any node that outputs VIDEO. Audio is dropped; the depth video is silent on purpose. |
| `target` | `seedance` · `h3` · `wan` · `none` | `seedance` | Which model the output is for. Sets **fps** and the **frame-size multiple**, both by the model's published rules: |
| | | | `seedance` — 24 fps, width and height multiples of 16 (Seedance 2.0 / 2.5 reference video). |
| | | | `h3` — 24 fps, multiples of 32 (MiniMax H3 reference video and Fun ControlNet). |
| | | | `wan` — 16 fps, multiples of 16 (Wan 2.1 VACE). |
| | | | `none` — keep the source fps, only make dimensions even. |
| | | | Frames are picked **by timestamp**, so 30 → 24 fps is really 24 (an integer stride would silently stay at 30). fps is never raised. Sizes are **cropped** to the multiple around the centre, never padded — a black border would read to the generator as a far wall. |
| `quality` | `fast` · `full` | `fast` | The resolution the depth model works at. Output size is **unchanged** either way; this is the size of the picture the model looks at. |
| | | | `fast` — 644×364 for 16:9, 364×644 for 9:16, 490×364 for 4:3. ~3 GB VRAM. **Recommended**: large shapes, positions and motion are identical to `full`. |
| | | | `full` — 924×518 / 518×924 / 686×518. ~11 GB VRAM, 2.5× slower. Sharper fine silhouettes (a loose strand of hair stays separate from the cheek). Measured against `fast` on the same 294-frame clip: mean difference 5.3 of 255 grey levels, 95 % of pixels within 15, edge energy −4.5 %. |
| `invert` | BOOLEAN | `false` | Off: near is white — what depth ControlNets, Seedance and MiniMax H3 were trained on. On: far is white, for tools that want the opposite. |
| `clip_percent` | FLOAT 0–10 | `0` | Trims that many percent from each end of the depth range before scaling to 0–255. Use 0.5–1 when one very close object (a fist at the lens) turns everything else dark. 0 reproduces the model's raw range. |
| `gamma` | FLOAT 0.2–3 | `1.0` | >1 darkens the mid-tones, spreading the near range out; <1 brightens them. 1 is linear. Rarely needed. |
| `max_side` | INT | `0` | Cap on the output's longer side, 0 = source size. **MiniMax H3 reference videos work best small — use 320** (that is 320×176 for 16:9): a full-size grey silhouette starts to pull the generated character's face shape towards the reference; a small one carries the moves without the shape. |

**Outputs**

| name | type | |
|---|---|---|
| `depth_video` | VIDEO | Connect to a reference-video input or Save Video. |
| `depth_frames` | IMAGE | The same frames as an IMAGE batch, for ControlNet inputs or Preview Image. |
| `fps` | FLOAT | The output fps after the preset (e.g. 24.0). Feed it to Create Video if you rebuild the video yourself. |

Warnings you may see in the console: `clip is 18.0s; seedance accepts <= 15s` — the node still runs, but the API will reject the file; trim the clip first.

### ReShot Depth Map

**Inputs**

| name | type | default | what it does |
|---|---|---|---|
| `images` | IMAGE | — | Frames as a batch `[T, H, W, 3]`, e.g. from Get Video Components, VHS Load Video, or Load Image (batch). |
| `quality` | `fast` · `full` | `fast` | As above. |
| `fit_to` | `none` · `h3` · `seedance` · `wan` | `none` | Center-crop to that model's frame-size multiple (h3 ×32, seedance/wan ×16). fps is not touched here — frames go out one for one. |
| `invert` / `clip_percent` / `gamma` | | | As above. |

**Output**: `depth` — IMAGE batch, same count and (unless `fit_to` cropped it) same size as the input, grey replicated to three channels so any node that takes IMAGE accepts it.

Both nodes share one loaded model per ComfyUI process. Depth is normalised over the **whole batch you pass** — so pass the whole clip at once, not chunks; chunks would each get their own scale and the grey would jump at the seams.

## Recipes

### A. Seedance 2.5 inside ComfyUI (API node)

ComfyUI's API nodes include **ByteDance Seedance 2.5 Reference to Video** (needs ComfyUI API credits). It has dynamic `reference_videos` inputs.

```
Load Video ─▶ ReShot Depth Video (target: seedance) ─depth_video─▶ Seedance 2.5 Reference to Video
                                                                    reference_videos: video 1
                                                                    prompt: see below
```

Prompt — point at the video and describe the people and the look:

```
参考@视频1的动作与运镜，顺序与视频保持一致。
一名穿深绿色丝绒旗袍的女子在狭窄的金属走廊里与三名黑衣守卫搏斗，冷蓝走廊光，红色警示灯，电影感。
```

(We have not run this node ourselves — it costs API credits; the input types match and the output of ReShot meets Seedance's reference-video spec: 24 fps, H.264, ≥ 407,696 pixels.)

### B. Seedance on the web (any ComfyUI)

Save the depth video (`ReShot Depth Video → Save Video`), then in the Seedance web UI upload it as a reference video and use the prompt above. Keep the clip ≤ 15 s (Seedance 2.0) — the `seedance` preset already sets 24 fps and the size multiple.

### C. MiniMax H3 reference video (open weights, in ComfyUI)

The core node **MiniMax H3 Reference to Video** takes `ref_videos` (dynamic VIDEO inputs) and `ref_images`.

```
Load Video ─▶ ReShot Depth Video (target: h3, max_side: 320) ─depth_video─▶ MiniMax H3 Reference to Video
Load Image (character sheet) ─────────────────────────────────image────────▶   ref_images: image 1
                                                                                ref_videos: video 1
```

MiniMax H3 reads its prompt in a fixed six-section format. The part that does the work is how `<Video 1>` is defined and what it may transfer:

```
<Subject 3> is the fight choreography and camera movement shown in <Video 1>, a grey depth map
in which near objects are white and far objects are black: one fighter leans on a corridor wall
in close-up, kicks off it to tear down a pipe, fights several opponents, is grabbed from behind
by the largest and throws him, slams the last one into a wall panel, wipes the mouth in close-up,
then walks away through a door past the fallen opponents.

<Subject 3>: attribute_transfer - every action, position, timing and camera move of <Video 1>
is transferred onto <Subject 1> and <Subject 2>; its grey depth look is not transferred.
```

Two things matter: **say in words what happens in the grey clip** (the model reads it far better with the narration), and **say the grey look is not to be copied**, or you may get a grey film back. Three complete prompts that produced the ReShot demo takes are in the main repo under [`docs/prompts/`](https://github.com/maosika-ai/reshot/tree/main/docs/prompts).

### D. MiniMax H3 Fun ControlNet (depth condition)

**Apply MiniMax H3 Fun ControlNet** takes `control_video` as an IMAGE batch. Use the frames path:

```
Load Video ─▶ Get Video Components ─images─▶ ReShot Depth Map (fit_to: h3) ─depth─▶ Apply MiniMax H3 Fun ControlNet: control_video
```

The Fun ControlNet Union weights and the depth conditioning are described at `https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union`. Keep `strength` near 1.0 to follow the blocking closely; lower it to let the model drift.

### E. Wan 2.1 VACE

**WanVaceToVideo** takes `control_video` as IMAGE.

```
Load Video ─▶ ReShot Depth Video (target: wan) ─depth_frames─▶ WanVaceToVideo: control_video
```

`target: wan` gives 16 fps and ×16 sizes, which is what VACE's samples use. (The wan values come from the docs; we have not verified this path end to end — reports welcome.)

## What transfers, and what doesn't

**Transfers:** who stands where, how big they are relative to each other, every move and its timing, the cuts, and the camera — push-ins, tracking, handheld shake.

**Doesn't:** faces (use a character sheet), clothes, lighting, colour, props in detail, and anything smaller than a hand. Those come from your prompt and reference images.

Things we learned making the demo:

- **Trim the reference to the shots you want before it goes in.** Everything in the clip gets copied, including the boring part at the end. Under 15 s for Seedance.
- **Small depth video for MiniMax H3** (`max_side` 320). See the table above for why.
- **Change the species, keep the size ratio.** A rabbit-vs-bear take works when the prompt says the bear is "about 1.3× the rabbit, never more than 1.5×". The depth map already says who is bigger; the prompt must not contradict it.
- **Plain clothes on extras, no logos.** Whatever the prompt leaves open, the model fills with text and badges.

## Performance

Measured 2026-09-12 with the same model and code the nodes use (the ReShot CLI), 294 frames of 736×1280:

| device | `quality` | model sees | VRAM peak | speed | host RAM |
|---|---|---|---|---|---|
| RTX 4090 | full | 518×924 | 11 GB | 62 ms/frame (12 s clip ≈ 18 s) | 3.9 GB |
| RTX 3080 Ti (12 GB) | full | 518×924 | 10.9 GB | 83 ms/frame | 3.5 GB |
| RTX 3080 Ti (12 GB) | **fast** | 364×644 | **3.0 GB** | **34 ms/frame** | 2.3 GB |
| 8 GB card (emulated by capping the 3080 Ti) | full | — | out of memory | — | — |
| 8 GB card (emulated) | fast | 364×644 | 3.0 GB | 34 ms/frame | 2.3 GB |
| Apple M2 Max | full | 518×924 | (unified) | ~500 ms/frame | ~4 GB |
| CPU | fast | — | — | ~1 s/frame | 2.3 GB |

Host RAM grows with clip length: about 2 GB + 224 MB per second of 720p at `fast`.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| Nodes don't appear; console says `IMPORT FAILED` for ComfyUI-ReShot | `reshot` not installed into ComfyUI's Python | Run the `pip install -r requirements.txt` line with the **same** python ComfyUI uses (`python_embeded\python.exe` on Windows portable). Check with `python -c "import reshot; print(reshot.__version__)"`. |
| `ReShot Depth Video` missing but `ReShot Depth Map` present | ComfyUI too old for the VIDEO type | Update ComfyUI, or use the frames path (Depth Map). |
| First run hangs or `could not obtain weights` | Hugging Face unreachable | Set `HF_ENDPOINT=https://hf-mirror.com` before starting ComfyUI (see First run), or copy the weights folder in by hand. |
| `CUDA out of memory … Tried to allocate 4.02 GiB` | `quality = full` on a card under 12 GB | Use `fast`. |
| Output is uniformly grey / flat | Static clip with almost no depth variation, or a huge hot pixel range | Try `clip_percent` 0.5–1; check the source actually has depth (a slide show won't). |
| The generated video follows the moves but the face drifts toward the reference | Depth video too big for MiniMax H3 | `max_side` 320. |
| The video model produced a grey film | Prompt didn't say the grey look is not to be copied | Add the retention line from Recipe C. |
| Seams / brightness jumps in the depth video | Frames were passed to Depth Map in chunks | Pass the whole clip in one batch; normalisation is per batch. |
| Very slow | Running on CPU | Check the console line `loading Video Depth Anything small on cpu`; make sure ComfyUI itself sees the GPU (`--cpu` flag off, correct torch build). |
| macOS: `Class AVFFrameReceiver is implemented in both …` at startup | pyav and opencv each bundle ffmpeg libs | Harmless warning; ignore. |
| `clip is 18.0s; seedance accepts <= 15s` | Reference too long for the API | Trim with Video Trim / VideoSlice before ReShot, or use `none`. |

Still stuck: open an issue with the console lines starting `ReShot:` and the ComfyUI version.

## FAQ

**Does it change my video's resolution?** No. `quality` is the size the depth model looks at; the depth video comes back at your source size (or `max_side` if you set it, or cropped to the model's multiple).

**Which quality should I use?** `fast`. Copying blocking and camera does not need the extra detail, and the demo takes on the ReShot page were made from a depth map shrunk to 320×176 before it reached the model. Use `full` for close-ups where thin silhouettes matter and you have 12 GB.

**Can I use it commercially?** Yes. The nodes, ReShot and the default model weights are Apache-2.0.

**Does it need the internet after the first run?** No. Weights are cached locally.

**Can I run it on a Mac?** Yes, on Apple Silicon (MPS) or CPU; expect minutes rather than seconds for a 12 s clip.

**Where do I get a character sheet for MiniMax H3?** Any consistent front/full-body image of your character. Maosika generates them as part of its pipeline; see the main repo's `docs/refs/` for the three used in the demo.

## Updating

ComfyUI Manager → **Update** on ComfyUI-ReShot, or `git pull` in `custom_nodes/ComfyUI-ReShot` followed by the `pip install -r requirements.txt` line again (that upgrades the `reshot` package from PyPI too). Restart ComfyUI.

To remove: delete the `custom_nodes/ComfyUI-ReShot` folder; optionally `pip uninstall reshot` and delete the weights folder from the Hugging Face cache.

## About Maosika 猫斯卡

[Maosika 猫斯卡](https://www.maosika.com) (www.maosika.com) is a professional AI video production system for short drama and short video: from a one-line idea to episodic script, character sheets, scene images, and every shot rendered with Seedance 2.0 / 2.5 and MiniMax H3 — a crew of digital specialists handling each step. ReShot is the depth-map step of that pipeline, open-sourced so anyone can copy a reference shot's staging and camera into their own AI video. To make AI short drama end to end, visit **https://www.maosika.com**.
