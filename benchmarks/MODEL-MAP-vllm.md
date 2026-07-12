# Ollama ↔ vLLM model map

Side-by-side runtime comparison only makes sense when both stacks run the
**same model family at a similar size and bit-width**. Ollama serves GGUF
(mostly `Q4_K_M`); vLLM serves Hugging Face weights (AWQ here). The quant
schemes are not bitwise identical — treat them as the same *class*
(~4-bit vs ~4-bit), not the same bytes.

| Ollama tag (RESULTS.md) | Params | Ollama quant | vLLM Hugging Face id | vLLM quant | On RTX 4080 16 GB |
| --- | --- | --- | --- | --- | --- |
| `qwen3.5:4b` | 4.7B | Q4_K_M | `cyankiwi/Qwen3.5-4B-AWQ-INT8-INT4` | AWQ ~4-bit | **default matrix** |
| `mistral:7b` | 7.2B | Q4_K_M | `solidrust/Mistral-7B-Instruct-v0.3-AWQ` | AWQ | **default matrix** |
| `llama3.1:8b` | 8.0B | Q4_K_M | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | AWQ-INT4 | **default matrix** |
| `ornith:9b` | 9.0B | Q4_K_M | `cyankiwi/Ornith-1.0-9B-AWQ-INT4` | AWQ-INT4 | **default matrix** |
| `qwen3.5:9b` | 9.7B | Q4_K_M | `sanskar003/Qwen3.5-9B-AWQ` | AWQ | **default matrix** |
| `qwen3.5:9b-q8_0` | 9.7B | Q8_0 | `RedHatAI/Qwen3.5-9B-FP8-dynamic` | FP8 | **OOM** (multimodal FP8) |
| `gemma3:12b` | 12.2B | Q4_K_M | `gaunernst/gemma-3-12b-it-int4-awq` | AWQ-INT4 | not in default (tight) |
| `qwen3.6:35b` | 36B MoE | Q4_K_M | `Qwen/Qwen3.6-35B-A3B-FP8` | FP8 | **no** (Ollama already CPU-offloads) |

### Hardware knobs used on this host

- `max-model-len=4096` — enough for the long-prompt bench (~1.2k tokens)
- `gpu-memory-utilization=0.80` — desktop often holds ~2 GiB already
- `--enforce-eager` — skips CUDA-graph capture (required for multimodal
  Qwen3.5-9B AWQ on 16 GB; small single-stream cost)

### Notes

- **Same prompts and metrics** as [`scripts/bench.sh`](../scripts/bench.sh).
- **`ornith:9b`** is [deepreinforce-ai/Ornith-1.0-9B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B)
  (agentic coding fine-tune on the Qwen3.5 family), not stock `qwen3.5:9b`.
- Qwen3.5 HF checkpoints are multimodal (`Qwen3_5ForConditionalGeneration`);
  their VRAM footprint is higher than the Ollama GGUF of the same parameter
  count. That is why the Q8↔FP8 row OOMs while the Q4↔AWQ row of the same
  base model fits.
- Pre-cache weights with [`scripts/pull-vllm-models.sh`](../scripts/pull-vllm-models.sh)
  so first start is not blocked by Hub rate limits.
