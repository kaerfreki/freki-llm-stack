#!/usr/bin/env bash
# Smoke test for the vLLM stack: waits for the OpenAI-compatible API, checks
# the served model, sends one real completion and verifies a non-empty answer.
#
#   VLLM_URL      endpoint to test          (default http://localhost:8000)
#   VLLM_MODEL    expected model id         (default: whatever /v1/models reports)
#   SMOKE_TIMEOUT seconds to wait for the API to come up (default 900)
#                 First start downloads weights into the HF cache volume and
#                 can take several minutes; subsequent starts are much faster.
#
# Default compose model is cyankiwi/Qwen3.5-4B-AWQ-INT8-INT4 — the ~4-bit
# counterpart of Ollama's qwen3.5:4b (see benchmarks/MODEL-MAP-vllm.md).
#
# Requires: curl, jq
set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8000}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-900}"

for dep in curl jq; do
  command -v "$dep" >/dev/null || { echo "✗ missing dependency: $dep"; exit 1; }
done

echo "→ Waiting for vLLM at ${VLLM_URL} (max ${SMOKE_TIMEOUT}s)..."
up=0
for ((i = 0; i < SMOKE_TIMEOUT; i++)); do
  if curl -sf "${VLLM_URL}/health" >/dev/null 2>&1; then
    up=1
    break
  fi
  sleep 1
done
if [[ $up -ne 1 ]]; then
  echo "✗ vLLM did not become healthy within ${SMOKE_TIMEOUT}s"
  echo "  Check: docker logs freki-vllm"
  exit 1
fi
echo "✓ vLLM /health is OK"

served=$(curl -sf "${VLLM_URL}/v1/models" | jq -r '.data[0].id // empty')
if [[ -z "$served" ]]; then
  echo "✗ /v1/models returned no model id"
  exit 1
fi
echo "✓ serving model: ${served}"

if [[ -n "${VLLM_MODEL:-}" && "$VLLM_MODEL" != "$served" ]]; then
  echo "✗ expected model '${VLLM_MODEL}' but server reports '${served}'"
  exit 1
fi

echo "→ Asking ${served} for a completion..."
response=$(curl -sf "${VLLM_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg m "$served" '{
    model: $m,
    prompt: "Reply with one short sentence: what is the capital of Norway?",
    max_tokens: 64,
    temperature: 0,
    seed: 42
  }')")

text=$(jq -r '.choices[0].text // empty' <<<"$response")
if [[ -z "$text" ]]; then
  echo "✗ empty response from model:"
  jq . <<<"$response"
  exit 1
fi

one_line=$(tr -s '[:space:]' ' ' <<<"$text" | sed 's/^ //; s/ $//')
completion_tokens=$(jq -r '.usage.completion_tokens // 0' <<<"$response")
echo "✓ model answered: ${one_line}"
echo "✓ completion tokens: ${completion_tokens}"
echo "✓ smoke test passed"
