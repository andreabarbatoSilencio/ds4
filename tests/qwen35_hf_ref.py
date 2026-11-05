#!/usr/bin/env python3
"""Compare the dense Qwen3.5-architecture graph with Hugging Face.

ds4 runs a llama.cpp qwen35 GGUF through its CPU reference and the CUDA
graph (first-token test), dumping the GPU logits of every position; HF runs
the original safetensors in float32 on CPU over the same token ids.  Use a
BF16 GGUF for implementation parity: it carries the checkpoint's own values,
so what remains is accumulation order and the f16 KV cache.

The 27B does not fit HF on a 32 GB host; Qwen3.5-0.8B (16/16 GDN heads) and
Qwen3.5-4B (32/16, the grouped value heads llama.cpp reorders) cover the
layer types the 27B uses.  Requires numpy, torch and transformers >= 5.2:

  uv run --with numpy --with torch --with transformers tests/qwen35_hf_ref.py \\
      --hf /path/to/Qwen3.5-4B --gguf /path/to/Qwen3.5-4B-BF16.gguf
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

TEXT = ("Gated DeltaNet layers keep a recurrent state, while every fourth layer uses full "
        "attention. In 1887 the Eiffel Tower's foundations were laid; its height is 330 m. "
        "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\nÈ un bel giorno.")


def log_softmax(x):
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hf", required=True, help="HF checkpoint directory")
    ap.add_argument("--gguf", required=True, help="llama.cpp qwen35 GGUF of the same checkpoint")
    ap.add_argument("--ds4", default="./ds4")
    ap.add_argument("--max-kl", type=float, default=1e-5, help="largest per-position KL(hf||ds4)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ids = AutoTokenizer.from_pretrained(args.hf)(TEXT, add_special_tokens=False)["input_ids"]
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "logits.f32")
        # one token per forward: every position then carries GPU logits
        env = dict(os.environ, DS4_QWEN4_GPU="1", DS4_QWEN4_GPU_ONLY="1", DS4_QWEN4_GPU_CHUNK="1",
                   DS4_QWEN4_FT_TOKENS=",".join(map(str, ids)), DS4_QWEN4_FT_OUT=out)
        r = subprocess.run([args.ds4, "-m", args.gguf, "--cuda", "--first-token-test", "-p", "x"],
                           env=env, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("ds4 failed:\n" + r.stderr[-4000:])
        got = np.fromfile(out, dtype=np.float32).reshape(len(ids), -1)

    model = AutoModelForCausalLM.from_pretrained(args.hf, dtype=torch.float32).eval()
    with torch.no_grad():
        ref = model(torch.tensor([ids])).logits[0].float().numpy()[:, :got.shape[1]]

    lp, lq = log_softmax(ref.astype(np.float64)), log_softmax(got.astype(np.float64))
    kl = (np.exp(lp) * (lp - lq)).sum(-1)
    agree = int((ref.argmax(-1) == got.argmax(-1)).sum())
    print(f"{len(ids)} positions: KL(hf||ds4) mean {kl.mean():.2e} max {kl.max():.2e}, "
          f"top1 {agree}/{len(ids)}, max|diff| {np.abs(ref - got).max():.4f}")
    if kl.max() > args.max_kl or agree != len(ids):
        sys.exit("FAIL: ds4 diverges from the HF reference")
    print("PASS")


if __name__ == "__main__":
    main()
