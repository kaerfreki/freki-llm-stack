#!/usr/bin/env bash
# Smoke test for the Ollama stack: waits for the API, pulls the model if
# missing, sends one real completion and verifies a non-empty answer.
#
#   OLLAMA_URL    endpoint to test          (default http://localhost:11434)
#   OLLAMA_MODEL  model to exercise         (default qwen3.5:9b)
#   SMOKE_TIMEOUT seconds to wait for the API to come up (default 60)
#
# Requires: curl, jq
set -euo pipefail

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.5:9b}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-60}"

for dep in curl jq; do
  command -v "$dep" >/dev/null || { echo "✗ missing dependency: $dep"; exit 1; }
done

echo "→ Waiting for Ollama at ${OLLAMA_URL} (max ${SMOKE_TIMEOUT}s)..."
up=0
for ((i = 0; i < SMOKE_TIMEOUT; i++)); do
  if curl -sf "${OLLAMA_URL}/api/version" >/dev/null 2>&1; then
    up=1
    break
  fi
  sleep 1
done
if [[ $up -ne 1 ]]; then
  echo "✗ Ollama did not answer within ${SMOKE_TIMEOUT}s"
  exit 1
fi
echo "✓ Ollama $(curl -sf "${OLLAMA_URL}/api/version" | jq -r .version) is up"

if ! curl -sf "${OLLAMA_URL}/api/tags" \
    | jq -e --arg m "$OLLAMA_MODEL" '.models[]? | select(.name == $m)' >/dev/null; then
  echo "→ Pulling ${OLLAMA_MODEL} (first run only, this can take a while)..."
  curl -sf -N "${OLLAMA_URL}/api/pull" -d "{\"model\": \"${OLLAMA_MODEL}\"}" \
    | jq -r 'select(.status) | .status' | uniq
  curl -sf "${OLLAMA_URL}/api/tags" \
    | jq -e --arg m "$OLLAMA_MODEL" '.models[]? | select(.name == $m)' >/dev/null \
    || { echo "✗ pull of ${OLLAMA_MODEL} failed"; exit 1; }
fi

echo "→ Asking ${OLLAMA_MODEL} for a completion..."
response=$(curl -sf "${OLLAMA_URL}/api/generate" -d "{
  \"model\": \"${OLLAMA_MODEL}\",
  \"prompt\": \"Reply with one short sentence: what is the capital of Norway?\",
  \"stream\": false
}")

text=$(jq -r '.response // empty' <<<"$response")
if [[ -z "$text" ]]; then
  echo "✗ empty response from model:"
  jq . <<<"$response"
  exit 1
fi

tokens_per_s=$(jq -r 'if .eval_duration > 0 then (.eval_count / (.eval_duration / 1e9) * 10 | round / 10) else "n/a" end' <<<"$response")
echo "✓ model answered: ${text}"
echo "✓ generation speed: ${tokens_per_s} tokens/s"
echo "✓ smoke test passed"
