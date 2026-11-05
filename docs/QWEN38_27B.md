# Qwen3.8 27B

[Back to README](../README.md)

Qwen3.8-27B is a dense model with the Qwen3.5 architecture: 64 layers
repeating three Gated DeltaNet layers and one gated GQA attention layer
(24 query and 4 KV heads), each followed by a SwiGLU FFN on a plain
pre-norm residual. Hidden size is 5120, the FFN is 17408 wide and the
native context is 262144 tokens. It is the Flash Next trunk without its
hyper-connections, n-grams, sparse attention and experts, and it shares
Flash Next's tokenizer and chat template byte for byte, so the CLI, native
agent, server, tool calls and thinking controls behave the same.

DwarfStar reads llama.cpp's `qwen35` GGUF directly. The widths come from the
GGUF, so the Qwen3.5 and 3.6 dense sizes load as well; the regression checks
below use Qwen3.5-0.8B and 4B. Qwen3.8-27B is the supported model.

## Download and run

CUDA only, single GPU:

```sh
make cuda-generic
./download_model.sh qwen38-27b-q4km
./ds4 --ctx 8192
```

| Target | File | Weights |
| --- | --- | --- |
| `qwen38-27b-q4km` | 17.67 GiB | Q4_K FFN, Q6_K attention output and LM head, Q8_0 attention and GDN |
| `qwen38-27b-q8` | 26.63 GiB | Q8_0 |

Both come from [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF)
and update `ds4flash.gguf`. The default prefill chunk is 2048 rows, about
1 GiB of transients; `--prefill-chunk` changes it.

The server exposes `qwen3.8-27b`, `qwen3.8-27b-chat` and
`qwen3.8-27b-reasoner`. Disk KV checkpoints include the recurrent state,
as for Flash Next.

## Memory on discrete GPUs

A discrete GPU without pageable memory access caches the weights in VRAM,
leaving free the planned sessions' context plus 512 MiB; the rest is pinned
in host memory and read over PCIe. The startup line reports the split:

```text
ds4: CUDA caching up to 5.65 GiB of the 17.67 GiB model in VRAM (1.85 GiB kept free for sessions); ...
```

Decode speed is then set by the bytes left on the host side. On an
RTX 3060 Ti (8 GB, PCIe 3.0) with 32 GB RAM, `ds4-bench` with
`speed-bench/promessi_sposi.txt` measured on October 5, 2026:

| Model | Prefill at 2K/4K | Decode |
| --- | ---: | ---: |
| Qwen3.8-27B Q4_K_M, 5.65 GiB in VRAM | 141 / 137 t/s | 0.70 t/s |
| Qwen3.5-0.8B Q8_0, all in VRAM | 362 t/s (short prompt) | 267 t/s |

The same machine read every weight over PCIe before this placement
(0.52 t/s for the 27B, 14 t/s for the 0.8B). Unified-memory and HMM
systems keep their direct mapping. Models of the family that fit in VRAM
run at full GPU speed, as the 0.8B row shows.

## Limitations

The MTP block is not implemented (`--mtp` is refused). Metal is refused: it
lacks the K-quant projections and the indexer-free attention preparation.
SSD streaming, tensor parallelism and pipeline execution are not supported,
and CUDA decodes server sessions in order. Vision is untested.

## Directional steering

Steering files are `64 x 5120` f32 (`--profile qwen3.8-27b` in
`dir-steering/tools/build_direction.py`). Unlike Flash Next, both edits act
on a block output before it joins the residual, as on DeepSeek and GLM, so
a negative FFN scale does not compound through the later layers.

## Validation

```sh
make test-qwen4-cuda
uv run --with numpy --with torch --with transformers tests/qwen35_hf_ref.py \
  --hf /path/to/Qwen3.5-4B --gguf /path/to/Qwen3.5-4B-BF16.gguf
DS4_QWEN4_GPU=1 ./ds4 -m gguf/Qwen3.8-27B-Q4_K_M.gguf --first-token-test \
  -p "The capital of France is Paris, and the capital of Italy is"
```

The kernel tests cover Q4_K and Q6_K projections at 1 to 40 rows and the
indexer-free attention preparation. The HF comparison passes BF16 0.8B and
4B checkpoints with a mean KL below 5e-8 and identical top tokens; it needs
the HF checkpoint in float32 RAM. The last command compares the CUDA graph
with the CPU reference on the 27B itself, at about 15 seconds per token for
the reference. On a 13-token sentence the worst logit difference was 0.002
with identical top tokens.
