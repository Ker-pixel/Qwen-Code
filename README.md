# Qwen 2.5 Coder CLI (MPS Accelerated)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A high-performance, visually polished, offline-first CLI for the **Qwen 2.5 Coder 1.5B** model. This tool is specifically engineered for **Apple Silicon (M1/M2/M3)** Macs, leveraging Metal Performance Shaders (MPS) for near-instant code generation and efficient memory handling.

## ✨ Key Features

* **Apple Silicon Optimized**: Uses `torch.mps` with custom memory watermarking (`HIGH_WATERMARK_RATIO=0.5`) to ensure stability on shared memory systems.
* **Offline Operation**: Downloads weights once (~3GB). Subsequent runs are strictly local—no data leaves your machine.
* **Senior Engineer Persona**: Default system prompt tuned for concise, optimized software engineering tasks.
* **Context Management**: Intelligent sliding window history (10 turns) to maintain speed and accuracy.
* **Modern UI**: ANSI-powered interface with real-time streaming, status bars, and loading spinners.
* **Memory Efficiency**: Automatic cache clearing and explicit object deletion after every inference cycle.

## 🚀 Quick Start

### Prerequisites
- **macOS**: Required for MPS hardware acceleration.
- **Python**: 3.9 or higher.
- **RAM**: 8GB minimum (16GB+ recommended).

### Installation

1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/qwen-coder-cli.git](https://github.com/yourusername/qwen-coder-cli.git)
   cd qwen-coder-cli

2. ** Set up Virtual Enviroment
   python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate

3. ** Install Dependencies
   pip install -r requirements.txt

### Running the CLI
Simply execute the script to start chatting:
  python main.py

  Note: On the first run, the script will download the model weights (~3GB) from Hugging Face into     the qwen_weights/ directory.

### Custom System Prompts

You can override the default persona via the CLI:

  python main.py --system "You are a specialized Rust security auditor. Focus on memory safety and edge cases."

### Technical Details

The script manages Apple Silicon's unified memory by setting specific environment variables before PyTorch initializes:

* PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5: Limits the model to 50% of system RAM to prevent OS swapping. 

* torch.mps.empty_cache(): Triggered after every response to keep the GPU memory footprint lean.

### License 

This project is licensed under the Apache License 2.0. See the LICENSE file for the full text.

### Acknowledgements

* [Qwen Team](https://huggingface.co/Qwen) for the Qwen 2.5 Coder 1.5B model.
* [Hugging Face](https://huggingface.co) for the Transformers and Accelerate libraries
