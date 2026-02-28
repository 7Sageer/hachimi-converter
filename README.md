# hachimi-converter

把任意音乐转换为哈基米风格的端到端音频风格迁移实验。

## 原理

输入原曲 mel spectrogram → U-Net 转换 → 输出哈基米风格 mel spectrogram → vocoder 还原音频

训练数据：B站上的"原版-哈基米版"配对音频，通过 chroma cross-correlation 自动对齐。

## 使用

```bash
# 1. 下载配对数据
python scripts/download.py

# 2. 对齐 & 切片
python scripts/slice_v2.py

# 3. 训练
python scripts/train.py

# 4. 推理
python scripts/inference.py input.wav output.wav 15
```

## 依赖

- PyTorch + torchaudio
- librosa, soundfile, matplotlib
- yt-dlp, ffmpeg

## 当前状态

- v0.1: 5首歌142个配对片段，base_ch=16 U-Net，Griffin-Lim vocoder
- 能听出"神韵"，但音质受限于 Griffin-Lim

## TODO

- [ ] 换 HiFi-GAN vocoder
- [ ] 扩充数据到 20-30 首
- [ ] 增大模型 + GPU 训练
