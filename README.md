# hachimi-converter

把任意音乐转换为哈基米风格的端到端音频风格迁移实验。

## 原理

输入原曲 mel spectrogram → U-Net 转换 → 输出哈基米风格 mel spectrogram → vocoder 还原音频

训练数据：B站上的"原版-哈基米版"配对音频，通过 chroma cross-correlation 自动对齐。

## 使用

```bash
# 1. 下载配对数据
python scripts/download.py

# 2. 下载 HiFi-GAN vocoder 权重
python scripts/download_vocoder.py

# 3. 对齐 & 切片
python scripts/slice_v2.py

# 4. 训练
python scripts/train.py

# 5. 推理
python scripts/inference.py input.wav output.wav 15
```

## 依赖

- PyTorch + torchaudio
- librosa, soundfile, matplotlib
- yt-dlp, ffmpeg
- gdown (vocoder 下载)

## 当前状态

- v0.2: HiFi-GAN UNIVERSAL_V1 vocoder，N_MELS=80，base_ch=32 U-Net
- mel 参数对齐 HiFi-GAN (N_FFT=1024, HOP=256, FMAX=8000)
- 需要用新参数重训 U-Net

## TODO

- [ ] 扩充数据到 20-30 首
- [ ] GPU 训练
- [x] 换 HiFi-GAN vocoder
