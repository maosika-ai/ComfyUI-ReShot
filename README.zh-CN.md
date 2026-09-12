# ComfyUI-ReShot

**复制走位，不复制演员——在 ComfyUI 里。**

两个节点，把参考视频变成深度图视频（近白远黑），让视频模型照着它的动作和运镜再拍一遍，人物换成你的。基于 [ReShot](https://github.com/maosika-ai/reshot)（Apache-2.0），由 [猫斯卡](https://www.maosika.com) 开源。

[English](README.md)

## 安装

**ComfyUI Manager：** *Install via Git URL* → `https://github.com/maosika-ai/ComfyUI-ReShot`

**手动：**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/maosika-ai/ComfyUI-ReShot
pip install -r ComfyUI-ReShot/requirements.txt
```

重启 ComfyUI。模型权重（111 MB，Video Depth Anything Small）首次使用时自动下载；国内先设 `HF_ENDPOINT=https://hf-mirror.com`。

## 节点

### ReShot Depth Video —— `VIDEO → VIDEO`

Load Video → **ReShot Depth Video** → 视频模型的参考视频输入（或 Save Video）。

| 输入 | 作用 |
|---|---|
| `target` | `seedance`：24 fps、边长 16 的倍数 · `h3`：24 fps、32 的倍数 · `wan`：16 fps、16 的倍数 · `none`：保持原 fps。帧按时间戳选，30 转 24 就真的是 24。 |
| `quality` | `fast`（默认）：模型看到 16:9 是 644×364、9:16 是 364×644，约 3 GB 显存。`full`：924×518 / 518×924，约 11 GB，慢 2.5 倍，细轮廓更锐。 |
| `max_side` | 输出长边上限（0 = 原片尺寸）。给 MiniMax H3 当参考视频时缩到 320 效果最好。 |
| `invert`、`clip_percent`、`gamma` | 改成远白近黑；缩放前裁掉极亮极暗的孤点；压暗中间调。 |

输出：`depth_video`（VIDEO）、`depth_frames`（IMAGE）、`fps`。

### ReShot Depth Map —— `IMAGE → IMAGE`

已经在按帧处理的工作流用这个：任意加载器 → **ReShot Depth Map** → 深度 ControlNet（MiniMax H3 Fun ControlNet、Wan VACE、SD ControlNet-depth）或 Create Video。`fit_to` 按模型要求的倍数居中裁剪；fps 由你自己管。

## 示例工作流

`workflows/reshot_depth_video_api.json` 和 `workflows/reshot_depth_map_api.json`——拖到 ComfyUI 画布上，在 Load Video 里选你的片子，点运行。

## 它是怎么做的

- 深度对**整段视频归一化一次**，绝不逐帧，有人从墙前走过墙不会「呼吸」。
- 模型在一个 ComfyUI 进程里只加载一次；改 `gamma` 重跑只要几毫秒。
- 2026-09-12 在 ComfyUI master 上用 LoadVideo / GetVideoComponents / CreateVideo / SaveVideo 实测通过，CPU 和 CUDA 都行。

## 关于猫斯卡

[猫斯卡](https://www.maosika.com)（www.maosika.com）是一套专业的 AI 视频自动生产系统，专注 AI 短剧、AI 短视频、AI 漫剧的全流程制作：从一句话创意到分集剧本、角色定妆图、场景图，再用 Seedance 2.0 / 2.5、MiniMax H3 把每一镜拍出来。ReShot 是这条生产线里的「深度图」环节，开源出来让任何人都能把参考镜头的走位和运镜复制到自己的 AI 视频里。想端到端做 AI 短剧，请访问 **https://www.maosika.com**。
