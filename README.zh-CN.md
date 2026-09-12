# ComfyUI-ReShot

**复制走位，不复制演员——在 ComfyUI 里。**

两个节点，把参考视频变成深度图视频（近白远黑），让视频模型照着它的动作和运镜再拍一遍——人物、衣服、画风都换成你的。基于 [ReShot](https://github.com/maosika-ai/reshot)（Apache-2.0），由 [猫斯卡](https://www.maosika.com) 开源。

[English](README.md)

1. [它做什么](#它做什么)
2. [需要什么](#需要什么)
3. [安装](#安装)
4. [第一次运行](#第一次运行)
5. [三步上手](#三步上手)
6. [节点参数详解](#节点参数详解)
7. [接法示例](#接法示例)——ComfyUI 里的 Seedance 2.5 · Seedance 网页 · MiniMax H3 参考视频 · MiniMax H3 Fun ControlNet · Wan VACE
8. [能复刻什么，不能复刻什么](#能复刻什么不能复刻什么)
9. [性能实测](#性能实测)
10. [排错](#排错)
11. [常见问题](#常见问题)
12. [更新与卸载](#更新与卸载)

---

## 它做什么

你看中了一段片子，那场打戏、那段舞、那个运镜正是你想要的。直接把片子当参考视频喂给视频模型，它会把脸、衣服、画风连同动作一起抄走；片子里是真人，平台审核可能直接拒收。改用文字描述动作，模型每次给你的打法都不一样。

ReShot 只留下你要的那部分。它用视频深度模型把片子扫一遍，写出一段灰片：近处白、远处黑、每一帧和上一帧严丝合缝。谁站在哪、谁大谁小、怎么动、镜头怎么走都保留；脸、服装、光线、画风全部去掉。把灰片当参考视频交给模型，提示词里写人物和画风。

这个包把它做成了 ComfyUI 的两个节点：

| 节点 | 进 → 出 | 什么时候用 |
|---|---|---|
| **ReShot Depth Video** | `VIDEO → VIDEO`（另给帧序列和 fps） | 你有 Load Video 节点，下游模型收参考**视频**——Seedance 2.5 Reference to Video、MiniMax H3 Reference to Video——或者你只想 Save Video 存成文件拿去别处上传。它会替你按那家模型的 fps 和尺寸规则处理。 |
| **ReShot Depth Map** | `IMAGE → IMAGE` | 你的工作流本来就按帧走——深度 ControlNet（MiniMax H3 Fun ControlNet、Wan VACE 的 `control_video`、SD ControlNet-depth）都收 IMAGE 批。 |

严格地说：单目视频深度估计，模型是 Video Depth Anything Small（字节跳动，CVPR 2025，Apache-2.0）。它以 32 帧为窗口重叠推理、逐帧预测相对逆深度并对齐；ReShot 对整段做一次归一化变成 8 位灰度。产出是深度图，不是 3D 模型。

## 需要什么

| | 最低 | 说明 |
|---|---|---|
| ComfyUI | 带原生 **Load Video / Save Video / Get Video Components** 节点的版本（2025 年 4 月之后） | `ReShot Depth Map`（IMAGE 进出）老版本也能用；`ReShot Depth Video` 需要 `VIDEO` 类型。 |
| Python | 3.10 – 3.12 | 跟你 ComfyUI 用的一致 |
| PyTorch | ≥ 2.1 | ComfyUI 能跑就有 |
| 显卡 | N 卡 **8 GB**（`quality = fast`），**12 GB**（`full`） | 实测数字见[性能实测](#性能实测)。Apple 芯片和纯 CPU 也能跑，慢。 |
| 磁盘 | 111 MB 模型权重 | 首次使用下载一次 |
| 内存 | 16 GB 能跑 720p 约 27 秒 | 峰值 ≈ 2 GB + 每秒 720p 约 224 MB |

节点**不需要** ffmpeg——解码和编码走 ComfyUI 自己的视频节点。

## 安装

### 用 ComfyUI Manager（推荐）

1. 打开 ComfyUI → **Manager** → **Custom Nodes Manager** → 右上角 **Install via Git URL**。
2. 粘贴 `https://github.com/maosika-ai/ComfyUI-ReShot` → **OK**。
3. Manager 会克隆仓库并执行 `pip install -r requirements.txt`，装上 `reshot` 包（缺的话连带 `huggingface_hub`、`opencv-python-headless`、`einops`、`easydict`）。
4. 按提示**重启 ComfyUI**。下次启动时导入列表里会出现 `custom_nodes/ComfyUI-ReShot`，带耗时、没有报错。

### 手动装

Linux / macOS，或你自己用 Python 装的 ComfyUI：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/maosika-ai/ComfyUI-ReShot
# 一定用 ComfyUI 自己那个 python：
python -m pip install -r ComfyUI-ReShot/requirements.txt
```

**Windows 便携版 ComfyUI**（带 `run_nvidia_gpu.bat` 的那个压缩包）——Python 在 `python_embeded` 里：

```bat
cd ComfyUI_windows_portable\ComfyUI\custom_nodes
git clone https://github.com/maosika-ai/ComfyUI-ReShot
cd ..\..
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-ReShot\requirements.txt
```

没装 git？在 GitHub 上下载仓库 zip，解压到 `custom_nodes/`（解压出来的文件夹里要直接有 `__init__.py`），再执行上面那行 `pip install`。

重启 ComfyUI。画布右键 → **Add Node** → **ReShot**，两个节点都在；或者双击画布输入 `reshot`。

### 或者让你的 AI 编程工具来装

把下面这段粘给 Claude Code、Codex、Cursor 或任何能在你电脑上执行命令的 AI 工具：

```
把 ComfyUI-ReShot 节点包装进我的 ComfyUI 并确认能加载。
先找到我的 ComfyUI 目录（找不到就问我），然后按 https://raw.githubusercontent.com/maosika-ai/ComfyUI-ReShot/main/README.zh-CN.md 做：
把 https://github.com/maosika-ai/ComfyUI-ReShot 克隆到 custom_nodes，用 ComfyUI 自己那个 python
（Windows 便携版是 python_embeded\python.exe）执行 pip install -r ComfyUI-ReShot/requirements.txt；
我在中国大陆，把 HF_ENDPOINT=https://hf-mirror.com 加进启动脚本。重启 ComfyUI，确认控制台里
custom_nodes/ComfyUI-ReShot 导入无报错、/object_info 里有 ReShotDepthVideo。没确认之前不要说完成。
```

## 第一次运行

ReShot 节点第一次执行时会从 Hugging Face 下载模型权重（`video_depth_anything_vits.pth`，111 MB）到 `~/.cache/huggingface/hub/models--depth-anything--Video-Depth-Anything-Small/`（Windows 在 `C:\Users\<你>\.cache\huggingface\hub\…`）。只下一次。控制台会显示下载进度，然后：

```
ReShot: 289 frames, model sees 644x364 (fast)
ReShot: loading Video Depth Anything small on cuda
```

**国内**不设镜像会卡死。在**启动 ComfyUI 之前**设环境变量：

- Linux / macOS：`export HF_ENDPOINT=https://hf-mirror.com`，然后在同一个终端里启动 ComfyUI。
- Windows 便携版：编辑 `run_nvidia_gpu.bat`，在 `python_embeded\python.exe` 那一行上面加一行 `set HF_ENDPOINT=https://hf-mirror.com`。

**离线机器**：在任何一台能联网的机器上从 `https://huggingface.co/depth-anything/Video-Depth-Anything-Small` 下载 `video_depth_anything_vits.pth`，把整个 `models--depth-anything--Video-Depth-Anything-Small` 文件夹拷到上面那个缓存路径。

模型在一个 ComfyUI 进程里只加载一次，之后的运行既不下载也不重新加载。

## 三步上手

把 `workflows/reshot_depth_video.json` 拖到画布上，一共三个节点：

```
Load Video ──VIDEO──▶ ReShot Depth Video ──depth_video──▶ Save Video
（参考片.mp4）          target: seedance                    reshot/depth_00001_.mp4
                       quality: fast
```

1. 在 **Load Video** 里点「选择要上传的文件」，选你的参考片（ffmpeg 认的都行；15 秒以内，见后面的经验）。
2. **ReShot Depth Video** 保持 `target = seedance`、`quality = fast`，除非你明确知道要改。
3. **运行**。控制台会打印帧数和模型工作的分辨率；深度视频落在 `ComfyUI/output/reshot/`。
4. 打开看：灰的、近白远黑、长度和构图跟你的片子一样。这个文件就是交给视频模型的东西。

第二个示例 `workflows/reshot_depth_map.json` 是按帧走的：Load Video → Get Video Components → **ReShot Depth Map** → Create Video → Save Video。下游是 ControlNet 时照这个改。

旁边的 `*_api.json` 是同样的图的 API 格式，给往 `/prompt` 发请求的脚本用。

## 节点参数详解

### ReShot Depth Video

**输入**

| 名称 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `video` | VIDEO | — | 来自 Load Video 或任何输出 VIDEO 的节点。音频会丢掉，深度视频故意是无声的。 |
| `target` | `seedance` · `h3` · `wan` · `none` | `seedance` | 输出给哪家模型。决定 **fps** 和**尺寸倍数**，都按各家公开的规则： |
| | | | `seedance`——24 fps，宽高都是 16 的倍数（Seedance 2.0 / 2.5 参考视频）。 |
| | | | `h3`——24 fps，32 的倍数（MiniMax H3 参考视频和 Fun ControlNet）。 |
| | | | `wan`——16 fps，16 的倍数（Wan 2.1 VACE）。 |
| | | | `none`——保持原 fps，只把宽高凑成偶数。 |
| | | | 帧**按时间戳**选，30 转 24 就真的是 24（按整数步长抽会悄悄停在 30）。fps 只降不升。尺寸按倍数**居中裁剪**，绝不补边——黑边会被生成模型当成远处的墙。 |
| `quality` | `fast` · `full` | `fast` | 深度模型工作的分辨率。**输出尺寸不变**，这只是模型看到的那张图的大小。 |
| | | | `fast`——16:9 是 644×364，9:16 是 364×644，4:3 是 490×364。约 3 GB 显存。**推荐**：大形状、位置、动作跟 `full` 一模一样。 |
| | | | `full`——924×518 / 518×924 / 686×518。约 11 GB 显存，慢 2.5 倍。细轮廓更锐（脸颊旁一缕散发不会糊进脸里）。同一段 294 帧的片上和 `fast` 对比：平均差 5.3 个灰阶（满量程 255），95% 像素差在 15 以内，边缘锐度低 4.5%。 |
| `invert` | BOOLEAN | `false` | 关：近白远黑——深度 ControlNet、Seedance、MiniMax H3 训练时用的就是这个方向。开：远白近黑，给要反着来的工具用。 |
| `clip_percent` | FLOAT 0–10 | `0` | 缩放到 0–255 之前，从深度范围两端各裁掉这个百分比。有一个贴着镜头的东西（比如冲到镜头前的拳头）把其他全压黑时，用 0.5–1。0 是模型原始范围。 |
| `gamma` | FLOAT 0.2–3 | `1.0` | >1 压暗中间调，把近处拉开；<1 提亮。1 是线性。很少需要动。 |
| `max_side` | INT | `0` | 输出长边上限，0 = 原片尺寸。**给 MiniMax H3 当参考视频时缩到 320 效果最好**（16:9 就是 320×176）：全尺寸的灰色人影会把生成角色的脸型往参考片里那个人上带；缩小后只带动作、不带脸型。 |

**输出**

| 名称 | 类型 | |
|---|---|---|
| `depth_video` | VIDEO | 接参考视频输入口或 Save Video。 |
| `depth_frames` | IMAGE | 同样的帧作为 IMAGE 批，接 ControlNet 或 Preview Image。 |
| `fps` | FLOAT | 预设处理后的 fps（比如 24.0）。自己重新拼视频时接给 Create Video。 |

控制台可能出现的警告：`clip is 18.0s; seedance accepts <= 15s`——节点照跑，但接口会拒收这个文件，先把片子剪短。

### ReShot Depth Map

**输入**

| 名称 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `images` | IMAGE | — | 帧批 `[T, H, W, 3]`，比如来自 Get Video Components、VHS Load Video 或批量 Load Image。 |
| `quality` | `fast` · `full` | `fast` | 同上。 |
| `fit_to` | `none` · `h3` · `seedance` · `wan` | `none` | 按那家模型的尺寸倍数居中裁剪（h3 是 32，seedance/wan 是 16）。这里不动 fps，帧一进一出。 |
| `invert` / `clip_percent` / `gamma` | | | 同上。 |

**输出**：`depth`——IMAGE 批，帧数不变，尺寸（除非 `fit_to` 裁了）不变，灰度复制成三通道，任何收 IMAGE 的节点都能接。

两个节点共用一个已加载的模型。深度对**你传进来的整批**做一次归一化——所以整段片一次传，别切块；切块各归一各的，接缝处灰度会跳。

## 接法示例

### A. ComfyUI 里直接调 Seedance 2.5（API 节点）

ComfyUI 的 API 节点里有 **ByteDance Seedance 2.5 Reference to Video**（需要 ComfyUI API 积分），带可增加的 `reference_videos` 输入。

```
Load Video ─▶ ReShot Depth Video (target: seedance) ─depth_video─▶ Seedance 2.5 Reference to Video
                                                                    reference_videos: video 1
                                                                    prompt: 见下
```

提示词——点名视频，再写人物和画风：

```
参考@视频1的动作与运镜，顺序与视频保持一致。
一名穿深绿色丝绒旗袍的女子在狭窄的金属走廊里与三名黑衣守卫搏斗，冷蓝走廊光，红色警示灯，电影感。
```

（这个节点我们自己没跑过——要花 API 积分；输入类型对得上，ReShot 的输出符合 Seedance 参考视频规格：24 fps、H.264、≥ 407,696 像素。）

### B. Seedance 网页（任何 ComfyUI）

先存文件（`ReShot Depth Video → Save Video`），到 Seedance 网页把它当参考视频上传，用上面的提示词。片子 ≤ 15 秒（Seedance 2.0）——`seedance` 预设已经把 24 fps 和尺寸倍数处理好了。

### C. MiniMax H3 参考视频（开源权重，ComfyUI 里）

核心节点 **MiniMax H3 Reference to Video** 收 `ref_videos`（可增加的 VIDEO 输入）和 `ref_images`。

```
Load Video ─▶ ReShot Depth Video (target: h3, max_side: 320) ─depth_video─▶ MiniMax H3 Reference to Video
Load Image（定妆图）───────────────────────────────────────────image────────▶   ref_images: image 1
                                                                                ref_videos: video 1
```

MiniMax H3 的提示词是固定的六段格式。真正起作用的是怎么定义 `<Video 1>`、允许它转移什么：

```
<Subject 3> is the fight choreography and camera movement shown in <Video 1>, a grey depth map
in which near objects are white and far objects are black: one fighter leans on a corridor wall
in close-up, kicks off it to tear down a pipe, fights several opponents, is grabbed from behind
by the largest and throws him, slams the last one into a wall panel, wipes the mouth in close-up,
then walks away through a door past the fallen opponents.

<Subject 3>: attribute_transfer - every action, position, timing and camera move of <Video 1>
is transferred onto <Subject 1> and <Subject 2>; its grey depth look is not transferred.
```

两个要点：**用文字把灰片里发生的事写一遍**（有了叙述模型读深度图准得多）；**明说灰色外观不要抄**，不然可能给你一部灰片。做出 ReShot 演示片的三条完整提示词在主仓库 [`docs/prompts/`](https://github.com/maosika-ai/reshot/tree/main/docs/prompts)。

### D. MiniMax H3 Fun ControlNet（深度条件）

**Apply MiniMax H3 Fun ControlNet** 的 `control_video` 收 IMAGE 批。走按帧的路：

```
Load Video ─▶ Get Video Components ─images─▶ ReShot Depth Map (fit_to: h3) ─depth─▶ Apply MiniMax H3 Fun ControlNet: control_video
```

Fun ControlNet Union 权重和深度条件说明见 `https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union`。`strength` 接近 1.0 就紧跟走位，调低让模型自由发挥。

### E. Wan 2.1 VACE

**WanVaceToVideo** 的 `control_video` 收 IMAGE。

```
Load Video ─▶ ReShot Depth Video (target: wan) ─depth_frames─▶ WanVaceToVideo: control_video
```

`target: wan` 给 16 fps 和 16 倍数的尺寸，和 VACE 官方示例一致。（wan 的数字来自文档，这条路我们没有端到端验证过——欢迎反馈。）

## 能复刻什么，不能复刻什么

**能：** 谁站在哪、彼此谁大谁小、每个动作和它的节奏、切镜、运镜——推进、跟拍、手持抖动。

**不能：** 脸（用定妆图）、衣服、光线、颜色、道具细节，以及比手还小的东西。这些全靠你的提示词和参考图。

做演示时踩出来的几条：

- **先把参考片剪到你要的那几个镜头再进来。** 片子里有什么就会复刻什么，包括结尾没用的那段。给 Seedance 的 15 秒以内。
- **给 MiniMax H3 的深度视频要小**（`max_side` 320），原因见上面的参数表。
- **换物种可以，体量比例别动。** 兔子打熊那段能成，是因为提示词写了熊「约为兔子的 1.3 倍，不超过 1.5 倍」。谁大谁小深度图已经定了，提示词不能跟它打架。
- **配角穿素色衣服，不留标志。** 提示词没写死的地方，模型会自己往上填字和徽章。

## 性能实测

2026-09-12 用节点同一套模型和代码（ReShot 命令行）实测，294 帧 736×1280：

| 设备 | `quality` | 模型看到 | 显存峰值 | 速度 | 内存 |
|---|---|---|---|---|---|
| RTX 4090 | full | 518×924 | 11 GB | 62 ms/帧（12 秒的片约 18 秒） | 3.9 GB |
| RTX 3080 Ti（12 GB） | full | 518×924 | 10.9 GB | 83 ms/帧 | 3.5 GB |
| RTX 3080 Ti（12 GB） | **fast** | 364×644 | **3.0 GB** | **34 ms/帧** | 2.3 GB |
| 8 GB 卡（用 3080 Ti 限显存模拟） | full | — | 显存溢出 | — | — |
| 8 GB 卡（模拟） | fast | 364×644 | 3.0 GB | 34 ms/帧 | 2.3 GB |
| Apple M2 Max | full | 518×924 | （统一内存） | 约 500 ms/帧 | 约 4 GB |
| 纯 CPU | fast | — | — | 约 1 秒/帧 | 2.3 GB |

内存随片长增长：`fast` 下大约 2 GB + 每秒 720p 约 224 MB。

## 排错

| 现象 | 原因 | 解决 |
|---|---|---|
| 找不到节点；控制台里 ComfyUI-ReShot 显示 `IMPORT FAILED` | `reshot` 没装进 ComfyUI 用的那个 Python | 用**同一个** python 执行 `pip install -r requirements.txt`（Windows 便携版是 `python_embeded\python.exe`）。用 `python -c "import reshot; print(reshot.__version__)"` 验证。 |
| 有 `ReShot Depth Map` 没有 `ReShot Depth Video` | ComfyUI 太老，没有 VIDEO 类型 | 更新 ComfyUI，或者走按帧的路（Depth Map）。 |
| 第一次运行卡住，或报 `could not obtain weights` | 连不上 Hugging Face | 启动 ComfyUI 前设 `HF_ENDPOINT=https://hf-mirror.com`（见「第一次运行」），或手动拷权重文件夹。 |
| `CUDA out of memory … Tried to allocate 4.02 GiB` | 12 GB 以下的卡用了 `quality = full` | 改 `fast`。 |
| 输出整片一个灰、没有层次 | 静止画面几乎没有深度变化，或者有极端亮点 | 试 `clip_percent` 0.5–1；确认原片真的有纵深（幻灯片没有）。 |
| 生成的片动作跟上了，脸却往参考片里的人偏 | 给 MiniMax H3 的深度视频太大 | `max_side` 320。 |
| 视频模型出了一部灰片 | 提示词没说灰色外观不要抄 | 加上接法 C 里那句 retention。 |
| 深度视频里有接缝 / 亮度跳变 | 帧是分块传给 Depth Map 的 | 整段一批传；归一化是按批做的。 |
| 特别慢 | 在 CPU 上跑 | 看控制台那行 `loading Video Depth Anything small on cpu`；确认 ComfyUI 本身认到了显卡（没加 `--cpu`，torch 是 GPU 版）。 |
| macOS 启动时报 `Class AVFFrameReceiver is implemented in both …` | pyav 和 opencv 各带了一份 ffmpeg 库 | 无害警告，忽略。 |
| `clip is 18.0s; seedance accepts <= 15s` | 参考片超过接口上限 | 在 ReShot 之前用 Video Trim / VideoSlice 剪短，或用 `none`。 |

还是不行：开 issue，附上控制台里 `ReShot:` 开头的行和 ComfyUI 版本。

## 常见问题

**会改我视频的分辨率吗？** 不会。`quality` 是深度模型看图的大小；深度视频按原片尺寸出来（设了 `max_side` 就按它，再按模型倍数裁一下）。

**该用哪个 quality？** `fast`。复制走位和运镜用不上多出来的细节，ReShot 主页上的演示片交给模型之前深度图还缩到了 320×176。特写里在意细轮廓、显卡有 12 GB，再用 `full`。

**能商用吗？** 能。节点、ReShot 和默认模型权重都是 Apache-2.0。

**第一次之后还要联网吗？** 不用，权重在本地缓存。

**Mac 上能跑吗？** 能，Apple 芯片（MPS）或 CPU；12 秒的片要按分钟算，不是秒。

**MiniMax H3 的定妆图哪来？** 任何一张你角色的正面 / 全身、前后一致的图都行。猫斯卡的流水线会自动生成；演示用的三张在主仓库 `docs/refs/`。

## 更新与卸载

ComfyUI Manager → 对 ComfyUI-ReShot 点 **Update**；或到 `custom_nodes/ComfyUI-ReShot` 里 `git pull`，再执行一次 `pip install -r requirements.txt`（顺带把 PyPI 上的 `reshot` 包升到最新）。重启 ComfyUI。

卸载：删掉 `custom_nodes/ComfyUI-ReShot` 文件夹；需要的话 `pip uninstall reshot`，再删 Hugging Face 缓存里的权重文件夹。

## 关于猫斯卡

[猫斯卡](https://www.maosika.com)（www.maosika.com）是一套专业的 AI 视频自动生产系统，专注 AI 短剧、AI 短视频、AI 漫剧的全流程制作：从一句话创意到分集剧本、角色定妆图、场景图，再用 Seedance 2.0 / 2.5、MiniMax H3 把每一镜拍出来——每个环节都由对应的数字专家接手。ReShot 是这条生产线里的「深度图」环节，开源出来让任何人都能把参考镜头的走位和运镜复制到自己的 AI 视频里。想端到端做 AI 短剧，请访问 **https://www.maosika.com**。
