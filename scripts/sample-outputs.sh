#!/usr/bin/env bash
# Generate the sample-output gallery: one markdown file per task under
# benchmarks/outputs/, containing every model's unedited answer to the same
# prompt (temperature 0, seed 42), for side-by-side quality comparison.
#
# Requires: curl, jq.
set -euo pipefail
export LC_ALL=C

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_DIR=$(dirname "$SCRIPT_DIR")/benchmarks

OLLAMA_URL=${OLLAMA_URL:-http://localhost:11434}
BENCH_MODELS=${BENCH_MODELS:-"qwen3.5:4b mistral:7b llama3.1:8b ornith:9b qwen3.5:9b qwen3.5:9b-q8_0 gemma3:12b qwen3.6:35b"}
# Generous: thinking models (qwen3.5/3.6) spend most of it on hidden reasoning.
# num_ctx must be raised too — Ollama's 4,096 default silently truncates the
# generation once prompt + thinking + answer no longer fit.
GALLERY_NUM_PREDICT=${GALLERY_NUM_PREDICT:-6144}
GALLERY_NUM_CTX=${GALLERY_NUM_CTX:-8192}

TASKS=${GALLERY_TASKS:-"summarize extract code reasoning trading legal rag"}

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

for cmd in curl jq; do command -v "$cmd" >/dev/null || die "$cmd is required"; done
curl -sf "$OLLAMA_URL/api/version" >/dev/null || die "Ollama API not reachable at $OLLAMA_URL"

task_file() {
    case $1 in
    summarize) echo "$BENCH_DIR/prompts/long.txt" ;;
    *) echo "$BENCH_DIR/tasks/$1.txt" ;;
    esac
}

task_blurb() {
    case $1 in
    summarize) echo "Summarize a ~1,200-token infrastructure report in one paragraph (same prompt as the long-prompt performance scenario)." ;;
    extract) echo "Extract and deduplicate the ERROR lines of a service log into a JSON array — instruction following and structured output." ;;
    code) echo "Implement a duration-string parser in Python with validation, docstring and examples." ;;
    reasoning) echo "A GPU memory-budget word problem with a single verifiable answer (93 sessions)." ;;
    trading) echo "Analyse a fixed 10-day price series — verifiable statistics (total return +4.9 %, max drawdown −6.8 %, worst day 7 at −2.7 %, SMA5 103.26 vs SMA10 102.99) plus a structured signal recommendation." ;;
    legal) echo "Extract the key terms of a contract excerpt into JSON — answer key: 24-month term, 90-day non-renewal notice, convenience exit after month 12 on 60 days' notice, 30-day cure period, liability capped at trailing-12-month fees except confidentiality/IP, French law, net 45." ;;
    rag) echo "Answer 6 questions strictly from a runbook; questions 3, 5 and 6 are NOT answerable from it (correct reply: \"Not in the document\") and are baited with prior-knowledge and decoy traps." ;;
    esac
}

# Prints the model's answer; strips <think> blocks some models emit.
generate() {
    local model=$1 prompt_file=$2 payload out
    # Model-default temperature (temp 0 sends thinking models into pathologically
    # verbose reasoning); fixed seed keeps a given run reproducible.
    local ctx=$GALLERY_NUM_CTX err ctx_note=""
    while :; do
        payload=$(jq -n --arg model "$model" --rawfile prompt "$prompt_file" \
            --argjson np "$GALLERY_NUM_PREDICT" --argjson ctx "$ctx" \
            '{model:$model, prompt:$prompt, stream:false,
              options:{seed:42, num_predict:$np, num_ctx:$ctx}}')
        out=$(curl -s "$OLLAMA_URL/api/generate" -d "$payload")
        err=$(jq -r '.error // empty' <<<"$out")
        [ -z "$err" ] && break
        # Whether a big model fits depends on what else (desktop included) is
        # using VRAM and host RAM at load time — CUDA OOM, a runner killed by
        # the kernel OOM killer ("unexpected EOF") or a load timing out under
        # memory thrash all mean the same thing: degrade the context window
        # instead of dying.
        if grep -qiE 'out of memory|unexpected EOF|timed out waiting' <<<"$err" &&
            [ "$ctx" -gt 2048 ]; then
            ctx=$((ctx / 2))
            log "$model: '$err' — retrying with num_ctx=$ctx"
            ctx_note="_[context window reduced to $ctx tokens after a memory-exhaustion error]_"
            continue
        fi
        die "$model: $err"
    done
    local answer
    answer=$(jq -r '.response' <<<"$out" |
        awk '/<think>/ {skip=1} /<\/think>/ {skip=0; next} !skip' |
        sed -e '/./,$!d')
    if [ -n "$answer" ]; then
        printf '%s\n' "$answer"
    else
        printf '_[the model produced no final answer — it was still reasoning when it hit the token budget]_\n'
    fi
    if [ -n "$ctx_note" ]; then
        printf '\n%s\n' "$ctx_note"
    fi
    local thinking_words
    thinking_words=$(jq -r '.thinking // ""' <<<"$out" | wc -w)
    if [ "$thinking_words" -gt 0 ]; then
        printf '\n_[the model produced ~%s words of hidden reasoning before this answer]_\n' "$thinking_words"
    fi
    if [ "$(jq -r '.done_reason' <<<"$out")" = "length" ]; then
        printf '\n_[output truncated at %s tokens]_\n' "$GALLERY_NUM_PREDICT"
    fi
}

mkdir -p "$BENCH_DIR/outputs"
version=$(curl -s "$OLLAMA_URL/api/version" | jq -r .version)

mkdir -p "$BENCH_DIR/outputs/.tmp"
trap 'rm -rf "$BENCH_DIR/outputs/.tmp"' EXIT

for task in $TASKS; do
    file=$(task_file "$task")
    [ -f "$file" ] || die "missing prompt file: $file"
    {
        echo "# Sample outputs — $task"
        echo
        echo "> Auto-generated by [\`scripts/sample-outputs.sh\`](../../scripts/sample-outputs.sh);"
        echo "> unedited model outputs at each model's default sampling settings"
        echo "> (\`seed=42\` for reproducibility), Ollama $version."
        echo "> $(task_blurb "$task")"
        echo
        echo "<details><summary>Full prompt</summary>"
        echo
        echo '```text'
        cat "$file"
        echo '```'
        echo
        echo "</details>"
    } >"$BENCH_DIR/outputs/.tmp/$task.md"
done

unload_all() {
    curl -s "$OLLAMA_URL/api/ps" | jq -r '.models[].name' | while read -r m; do
        curl -s "$OLLAMA_URL/api/generate" \
            -d "$(jq -n --arg m "$m" '{model:$m, keep_alive:0}')" >/dev/null
    done
}

# Model-major order so each model is loaded into VRAM only once.
for model in $BENCH_MODELS; do
    unload_all # co-resident models can OOM the big ones at num_ctx 8192
    for task in $TASKS; do
        log "$model: $task"
        {
            echo
            echo "## \`$model\`"
            echo
            generate "$model" "$(task_file "$task")"
        } >>"$BENCH_DIR/outputs/.tmp/$task.md"
    done
done

for task in $TASKS; do
    mv "$BENCH_DIR/outputs/.tmp/$task.md" "$BENCH_DIR/outputs/$task.md"
    log "wrote $BENCH_DIR/outputs/$task.md"
done
log "gallery done"
