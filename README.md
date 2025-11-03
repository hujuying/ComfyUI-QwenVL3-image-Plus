# ComfyUI-QwenVL3-image

[English](#english) | [中文](#中文)

---

<a name="中文"></a>
## 中文文档

### 📖 简介

- 一个为 ComfyUI 设计的自定义节点，集成了 Qwen3-VL-4B-Instruct-FP8 视觉语言模型，用于高效的图像理解和描述。支持灵活的提示词组合功能，提升图像描述的精准度和多样性。

### ✨ 主要特性

- 🚀 **高效 FP8 量化**：仅需约 10GB 显存
- 📦 **批量处理支持**：一次处理多张图片
- 💾 **智能内存管理**：可选模型保持加载，优化显存
- 🔧 **辅助工具链**：提供文本分割、列表处理等节点
- 📝 灵活提示词组合：支持预设提示词、自定义提示词及附加提示词三重输入，自动组合生成描述指令

### 📋 硬件要求

- **GPU**: NVIDIA RTX 3090 或更高（计算能力 ≥ 8.9）
- **显存**: ≥ 10GB
- **系统内存**: 8GB+

> ⚠️ **重要提示**: 此插件仅支持 FP8 量化模型，需要计算能力 8.9 或更高的 GPU。

### 🔧 安装方法

#### 使用 Git Clone（推荐）

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/yamanacn/ComfyUI-QwenVL3-image.git
cd ComfyUI-QwenVL3-image
pip install -r requirements.txt
```

#### 使用 ComfyUI Manager

1.  在 ComfyUI 中打开 Manager
2.  搜索 "QwenVL3"
3.  点击安装

### 📦 模型下载

模型会在首次使用时自动下载。你也可以从 HuggingFace 手动下载模型，并将其放置在 `ComfyUI/models/Qwen/` 目录下。

- **模型地址**: [Qwen/Qwen3-VL-4B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-FP8)

### 🎮 基础工作流

```
图片输入 → QwenVL3 Image (FP8) → 文本输出
```

对于批量处理，可连接 `Text Batch Splitter` 和 `List Selector` 节点来分别查看每张图片的描述。

### 🙏 致谢

- [Qwen Team](https://github.com/QwenLM/Qwen)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Hugging Face](https://huggingface.co/)

### 📄 许可证

本项目采用 MIT 许可证。

---

<a name="english"></a>
## English Documentation

### 📖 Introduction

A custom node for ComfyUI that integrates the **Qwen3-VL-4B-Instruct-FP8** vision-language model for efficient image understanding and description.

### ✨ Key Features

- 🚀 **Efficient FP8 Quantization**: Runs with only ~10GB VRAM
- 📦 **Batch Processing Support**: Process multiple images at once
- 💾 **Smart Memory Management**: Optional model persistence for optimized VRAM usage
- 🔧 **Auxiliary Toolchain**: Includes nodes for text splitting and list processing

### 📋 Hardware Requirements

- **GPU**: NVIDIA RTX 4090 or higher (Compute Capability ≥ 8.9)
- **VRAM**: ≥ 10GB
- **System RAM**: 8GB+

> ⚠️ **Important**: This plugin only supports FP8 quantized models, requiring GPUs with compute capability 8.9 or higher.

### 🔧 Installation

#### Git Clone (Recommended)

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/yamanacn/ComfyUI-QwenVL3-image.git
cd ComfyUI-QwenVL3-image
pip install -r requirements.txt
```

#### ComfyUI Manager

1.  Open Manager in ComfyUI
2.  Search for "QwenVL3"
3.  Click Install

### 📦 Model Download

The model is downloaded automatically on first use. You can also manually download it from HuggingFace and place it in the `ComfyUI/models/Qwen/` directory.

- **Model URL**: [Qwen/Qwen3-VL-4B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-FP8)

### 🎮 Basic Workflow

```
Image Input → QwenVL3 Image (FP8) → Text Output
```

For batch processing, connect the `Text Batch Splitter` and `List Selector` nodes to view descriptions for each image separately.

### 🙏 Credits

- [Qwen Team](https://github.com/QwenLM/Qwen)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Hugging Face](https://huggingface.co/)

### 📄 License

This project is licensed under the MIT License.

