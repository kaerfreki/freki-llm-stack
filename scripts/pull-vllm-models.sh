#!/usr/bin/env bash
# Pre-download the Hugging Face weights used by the vLLM benchmark into the
# compose volume (vllm_vllm-huggingface). Avoids first-start hangs from the
# unauthenticated Hub rate limit. Pairing table: benchmarks/MODEL-MAP-vllm.md.
#
#   ./scripts/pull-vllm-models.sh
#   BENCH_PAIRS="qwen3.5:4b=cyankiwi/Qwen3.5-4B-AWQ-INT8-INT4" ./scripts/pull-vllm-models.sh
#
# Requires: docker. Uses the pinned vllm/vllm-openai image (CPU-only pull job).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMPOSE_DIR=$(dirname "$SCRIPT_DIR")/compose/vllm
VLLM_IMAGE_TAG=${VLLM_IMAGE_TAG:-v0.25.0}
VOLUME_NAME=${VLLM_VOLUME:-vllm_vllm-huggingface}

BENCH_PAIRS=${BENCH_PAIRS:-"\
qwen3.5:4b=cyankiwi/Qwen3.5-4B-AWQ-INT8-INT4 \
mistral:7b=solidrust/Mistral-7B-Instruct-v0.3-AWQ \
llama3.1:8b=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
ornith:9b=cyankiwi/Ornith-1.0-9B-AWQ-INT4 \
qwen3.5:9b=sanskar003/Qwen3.5-9B-AWQ"}

# shellcheck disable=SC2206
pairs=($BENCH_PAIRS)
models=()
for pair in "${pairs[@]}"; do
  models+=("${pair#*=}")
done

# Ensure the named volume exists (create via a no-op compose config if needed)
if ! docker volume inspect "$VOLUME_NAME" >/dev/null 2>&1; then
  echo "→ creating volume $VOLUME_NAME via compose project"
  (cd "$COMPOSE_DIR" && docker compose up --no-start >/dev/null 2>&1 || true)
  docker volume inspect "$VOLUME_NAME" >/dev/null \
    || { echo "✗ volume $VOLUME_NAME not found; run: cd compose/vllm && docker compose up -d once"; exit 1; }
fi

echo "→ pulling ${#models[@]} model(s) into volume $VOLUME_NAME (image vllm/vllm-openai:${VLLM_IMAGE_TAG})"
# shellcheck disable=SC2086
docker run --rm --entrypoint bash \
  -v "${VOLUME_NAME}:/root/.cache/huggingface" \
  "vllm/vllm-openai:${VLLM_IMAGE_TAG}" \
  -lc "
set -e
python3 - <<'PY'
from huggingface_hub import snapshot_download
models = '''${models[*]}'''.split()
for m in models:
    print(f'=== {m} ===', flush=True)
    path = snapshot_download(m)
    print(f'OK -> {path}', flush=True)
print('all models cached', flush=True)
PY
"
echo "✓ pull-vllm-models done"
