import os
import sys
import time
import threading

# High watermark is the hard limit
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.5"
# Low watermark must be LESS than or equal to High
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.3"

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(BASE_PATH, "qwen_weights")
MAX_HISTORY_TURNS = 10

# ── ANSI palette ────────────────────────────────────────────────x
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"

RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"

BG_BLACK = "\033[40m"

def c(text, *styles):
    return "".join(styles) + str(text) + RESET

# ── Spinner ──────────────────────────────────────────────────────
class Spinner:
    FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

    def __init__(self, message):
        self.message = message
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r  {c(frame, CYAN, BOLD)}  {c(self.message, DIM)}  ")
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

# ── Helpers ──────────────────────────────────────────────────────
def setup_cli():
    parser = argparse.ArgumentParser(description="Qwen 2.5 Coder — offline CLI")
    parser.add_argument("--system", type=str,
                        default="You are a senior software engineer. Provide concise and optimized code.",
                        help="System prompt.")
    return parser.parse_args()

def is_model_downloaded(path):
    if not os.path.exists(path):
        return False
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith((".safetensors", ".bin", ".pt")):
                return True
    return False

def free_mps_memory(*tensors):
    for t in tensors:
        del t
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def ram_usage_gb():
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1_073_741_824 if sys.platform == "darwin" else 1_048_576
        return kb / divisor
    except Exception:
        return 0.0

def status_bar(device_str, offline):
    ram  = ram_usage_gb()
    mode = c(" OFFLINE ", BG_BLACK, GREEN, BOLD) if offline else c(" ONLINE ", BG_BLACK, YELLOW, BOLD)
    dev  = c(f" {device_str.upper()} ", BG_BLACK, CYAN, BOLD)
    mem  = c(f" RAM {ram:.1f} GB ", BG_BLACK, MAGENTA, BOLD)
    mdl  = c(f" {MODEL_ID.split('/')[-1]} ", BG_BLACK, DIM)
    print(f"\n{mode}{dev}{mem}{mdl}")

def print_banner():
    lines = [
        "  ██████╗ ██╗    ██╗███████╗███╗   ██╗",
        " ██╔═══██╗██║    ██║██╔════╝████╗  ██║",
        " ██║   ██║██║ █╗ ██║█████╗  ██╔██╗ ██║",
        " ██║▄▄ ██║██║███╗██║██╔══╝  ██║╚██╗██║",
        " ╚██████╔╝╚███╔███╔╝███████╗██║ ╚████║",
        "  ╚══▀▀═╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝",
    ]
    bar = c("━" * 42, DIM)
    print(f"\n{bar}")
    for line in lines:
        print(c(line, MAGENTA, BOLD))
    print(bar)
    print(c("  Qwen 2.5 Coder  ·  1.5B  ·  MPS Accelerated", DIM))
    print(c("  /clear  to reset context  ·  exit  to quit\n", DIM))

# ── Main ─────────────────────────────────────────────────────────
def main():
    args = setup_cli()

    os.system("clear")
    print_banner()

    if torch.backends.mps.is_available():
        device     = torch.device("mps")
        dtype      = torch.float16
        device_str = "MPS"
        print(c("  ✓ Metal Performance Shaders active", GREEN))
    else:
        device     = torch.device("cpu")
        dtype      = torch.float32
        device_str = "CPU"
        print(c("  ⚠ MPS not found — falling back to CPU", YELLOW))

    offline_mode = is_model_downloaded(LOCAL_DIR)

    if offline_mode:
        print(c("  ✓ Local weights found — strict offline mode", GREEN))
    else:
        print(c("  ↓ No local weights — downloading on first run (~3 GB)", YELLOW))

    print()
    with Spinner("Loading tokenizer …"):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID,
                cache_dir=LOCAL_DIR,
                trust_remote_code=True,
                local_files_only=offline_mode
            )
        except Exception as e:
            sys.stdout.write("\r\033[K")
            print(c(f"\n  ✗ Tokenizer load failed: {e}", RED))
            sys.exit(1)
    print(c("  ✓ Tokenizer ready", GREEN))

    with Spinner("Loading model weights into MPS …"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map={"": device},
                cache_dir=LOCAL_DIR,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                local_files_only=offline_mode
            )
        except Exception as e:
            sys.stdout.write("\r\033[K")
            print(c(f"\n  ✗ Model load failed: {e}", RED))
            sys.exit(1)
    print(c("  ✓ Model loaded", GREEN))

    status_bar(device_str, offline_mode)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    messages = [{"role": "system", "content": args.system}]

    # ── Inference loop ────────────────────────────────────────────
    while True:
        try:
            sys.stdout.write(f"\n{c('  ❯', CYAN, BOLD)} ")
            sys.stdout.flush()
            user_input = input()

            if user_input.lower() in ["exit", "quit"]:
                print(c("\n  Session ended.\n", DIM))
                break

            if user_input.lower() == "/clear":
                messages = [{"role": "system", "content": args.system}]
                print(c("  ✓ Context cleared", GREEN))
                continue

            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})

            if len(messages) > MAX_HISTORY_TURNS + 1:
                messages = [messages[0]] + messages[-MAX_HISTORY_TURNS:]
                print(c("  ↻ History trimmed", DIM))

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            cpu_inputs     = tokenizer([text], return_tensors="pt")
            input_ids      = cpu_inputs.input_ids.to(device)
            attention_mask = cpu_inputs.attention_mask.to(device)
            del cpu_inputs, text

            print(f"\n{c('  ╭─', DIM)} {c('Qwen', CYAN, BOLD)}\n{c('  │', DIM)} ", end="", flush=True)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    streamer=streamer,
                    temperature=0.2,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            print(c("  ╰─", DIM))

            prompt_len    = input_ids.shape[1]
            new_token_ids = output_ids[0][prompt_len:]
            response      = tokenizer.decode(new_token_ids, skip_special_tokens=True)

            del output_ids, input_ids, attention_mask, new_token_ids
            free_mps_memory()

            messages.append({"role": "assistant", "content": response})


        except KeyboardInterrupt:
            print(c("\n\n  Session ended.\n", DIM))
            break

if __name__ == "__main__":
    main()