# Qwen 2.5 Coder CLI

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)]()

An offline-first, visually polished CLI for **Qwen 2.5 Coder 1.5B**, built specifically for **Apple Silicon (M1/M2/M3/M4/M5)** Macs. Download the weights once, then run entirely locally — no internet required, no data leaves your machine.

---

## Features

- **Apple Silicon optimized** — runs via Metal Performance Shaders (MPS) with custom memory watermarking for stability on unified memory systems
- **Offline-first** — weights download once (~3 GB); every subsequent run is strictly local
- **RAM efficient** — explicit tensor cleanup and MPS cache flushing after every inference cycle keeps the footprint flat at ~4 GB
- **Sliding context window** — keeps the last 10 turns so long sessions don't balloon memory
- **Polished terminal UI** — ANSI status bar, braille spinner on load, live RAM readout after each response
- **Custom personas** — override the system prompt via `--system` flag

---

## Requirements

- **macOS** with Apple Silicon (MPS is not available on Intel Macs)
- **Python 3.9+**
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk**: ~3 GB free for model weights

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/qwen-code.git
cd qwen-code

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

The first run downloads the model weights (~3 GB) into `qwen_weights/`. Every run after that is fully offline — the script detects the weights and sets `local_files_only=True` automatically.

### Custom system prompt

```bash
python main.py --system "You are a Rust security auditor. Focus on memory safety and edge cases."
```

### In-session commands

| Command | Effect |
|---------|--------|
| `/clear` | Wipe conversation history, keep model loaded |
| `exit` / `quit` | End the session |

---

## Network-restricted regions

If HuggingFace is blocked or throttled in your region (common in Turkey and parts of Asia), set the mirror endpoint before the first run:

```bash
HF_ENDPOINT=https://hf-mirror.com python main.py
```

Or add it permanently to your shell profile:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Once the weights are downloaded, the mirror setting is no longer needed.

---

## Memory design

Qwen 2.5 Coder 1.5B at `float16` has a hard floor of ~3 GB just for weights. The script keeps total usage flat at ~4 GB through three mechanisms:

1. **Explicit tensor deletion** — CPU and MPS tensors are `del`'d immediately after use, before `empty_cache()` fires, so PyTorch can actually reclaim them
2. **History truncation before tokenizing** — the sliding window cuts old turns before the prompt is built, not after, so you never tokenize a prompt you're about to discard
3. **MPS watermarks** — `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5` caps MPS allocation at 60% of system RAM to prevent OS-level swapping

## Offline detection

The script uses `os.walk` (not `os.listdir`) to detect cached weights. HuggingFace nests model files several directories deep inside `qwen_weights/`, so a shallow directory check would always return `False` and attempt a network fetch even when the weights are fully cached.

---

## Project structure

```
qwen-code/
├── main.py            # CLI entry point
├── requirements.txt   # torch, transformers
├── qwen_weights/      # auto-created on first run (gitignored)
├── LICENSE
└── README.md
```

---

## Contributing

Bug reports and PRs are welcome. If you're adding support for a different model, the two things most likely to need updating are the `MODEL_ID` constant and the watermark ratios — larger models need a lower `HIGH_WATERMARK_RATIO` to stay stable.

---

## Acknowledgements

- [Qwen Team](https://huggingface.co/Qwen) for the Qwen 2.5 Coder 1.5B model
- [Hugging Face](https://huggingface.co) for the Transformers library
- [hf-mirror.com](https://hf-mirror.com) for the open HuggingFace mirror

---

## License

Licensed under the [Apache License 2.0](LICENSE).
