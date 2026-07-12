#!/usr/bin/env bash
# Context-window sweep for Ollama — how num_ctx trades VRAM, TTFT and tok/s.
#
#   bench-ctx.sh run             run the sweep, write CSV + RESULTS-ctx.md
#   bench-ctx.sh report <csv>    regenerate RESULTS-ctx.md from an existing CSV
#
# Default target is ornith:9b (agentic coding model used with OpenCode). The
# model card advertises 262k native context, but Ollama only reserves what
# you pass as options.num_ctx — the default is far smaller and is the usual
# reason OpenCode hits "context length exceeded" mid-session.
#
# Per num_ctx: unload → 1 warm-up (cold load with that ctx) → generation
# (short prompt) and prefill (synthetic prompt ≈ 75% of num_ctx) scenarios,
# BENCH_RUNS measured runs each, medians reported. VRAM sampled in flight.
#
# Requires: curl, jq, awk, nvidia-smi, python3.
set -euo pipefail
export LC_ALL=C

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_DIR=$(dirname "$SCRIPT_DIR")/benchmarks

OLLAMA_URL=${OLLAMA_URL:-http://localhost:11434}
BENCH_MODEL=${BENCH_MODEL:-ornith:9b}
# Sizes that matter for coding agents; stop early on OOM / API error.
BENCH_CTX_SIZES=${BENCH_CTX_SIZES:-"2048 4096 8192 16384 32768 65536 131072"}
BENCH_RUNS=${BENCH_RUNS:-3}
BENCH_NUM_PREDICT=${BENCH_NUM_PREDICT:-128}
BENCH_KEEP_ALIVE=${BENCH_KEEP_ALIVE:-15m}
# Fraction of num_ctx filled by the synthetic prefill prompt (leave room for output).
BENCH_PREFILL_FRAC=${BENCH_PREFILL_FRAC:-0.75}

CSV_HEADER="model,num_ctx,scenario,run,ttft_ms,gen_tps,prompt_tps,load_ms,eval_count,prompt_eval_count,vram_peak_mib,ram_used_peak_mib,split"

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

for cmd in curl jq awk nvidia-smi python3; do
    command -v "$cmd" >/dev/null || die "$cmd is required"
done

# --- sampling -----------------------------------------------------------

SAMPLER_PID=""
SAMPLER_FILE=""

sampler_start() {
    SAMPLER_FILE=$(mktemp)
    (
        while :; do
            vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
            ram=$(awk '/^MemAvailable/ {print int($2/1024)}' /proc/meminfo)
            echo "${vram:-0} ${ram:-0}"
            sleep 0.5
        done
    ) >"$SAMPLER_FILE" 2>/dev/null &
    SAMPLER_PID=$!
}

sampler_stop() {
    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
    awk -v total="$(awk '/^MemTotal/ {print int($2/1024)}' /proc/meminfo)" '
        { if ($1 > v) v = $1; used = total - $2; if (used > r) r = used }
        END { print v+0, r+0 }' "$SAMPLER_FILE"
    rm -f "$SAMPLER_FILE"
    SAMPLER_PID=""
}

cleanup() { [ -n "${SAMPLER_PID:-}" ] && kill "$SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# --- ollama helpers -----------------------------------------------------

unload_all() {
    curl -s "$OLLAMA_URL/api/ps" | jq -r '.models[].name // empty' | while read -r m; do
        [ -z "$m" ] && continue
        curl -s "$OLLAMA_URL/api/generate" \
            -d "$(jq -n --arg m "$m" '{model:$m, keep_alive:0}')" >/dev/null
    done
    sleep 3
}

model_gpu_split() {
    curl -s "$OLLAMA_URL/api/ps" | jq -r --arg m "$1" '
        .models[]? | select(.name==$m) |
        (.size_vram / .size * 100 | round) as $g |
        if $g >= 100 then "100% GPU" else "\($g)% GPU / \(100-$g)% CPU" end
        // "n/a"'
}

model_reported_ctx() {
    curl -s "$OLLAMA_URL/api/ps" | jq -r --arg m "$1" '
        .models[]? | select(.name==$m) | .context_length // empty'
}

# Build a prompt file whose rough token count is ~target (English ≈ 4 chars/token).
# Actual prompt_eval_count from Ollama is what the report uses.
make_prefill_prompt() {
    local target_tokens=$1 out=$2
    python3 - "$target_tokens" "$out" <<'PY'
import sys
target = int(sys.argv[1])
out = sys.argv[2]
# ~4 chars/token; include a real task at the end so the model does work.
unit = (
    "Section %d. Infrastructure note: self-hosted inference on a single NVIDIA GPU "
    "requires sizing VRAM for both weights and the KV cache. Larger num_ctx values "
    "reserve more memory at load time even before the prompt is filled. "
)
# Leave headroom for instruction wrapper and a short answer.
body_tokens = max(64, target - 80)
chars = body_tokens * 4
chunks = []
n = 0
while sum(len(c) for c in chunks) < chars:
    n += 1
    chunks.append(unit % n)
body = "".join(chunks)[:chars]
prompt = (
    "You are helping size context windows for a coding agent.\n\n"
    "CONTEXT DUMP (ignore detail, scan only):\n"
    f"{body}\n\n"
    "Task: in one short sentence, state whether larger context windows cost more "
    "VRAM at load time even when the prompt is short. Answer only that sentence."
)
open(out, "w", encoding="utf-8").write(prompt)
print(len(prompt), flush=True)
PY
}

# --- one measured request -----------------------------------------------
# Prints: ttft_ms gen_tps prompt_tps load_ms eval_count prompt_eval_count

run_once() {
    local model=$1 prompt_file=$2 num_predict=$3 num_ctx=$4 tag=$5
    local payload_file start now ttft_ms="" last="" line

    # Write the JSON body to a file — large prefills exceed ARG_MAX if passed
    # on the curl command line via -d "$payload".
    payload_file=$(mktemp)
    jq -n --arg model "$model" --arg tag "$tag" \
        --rawfile prompt "$prompt_file" \
        --argjson np "$num_predict" --argjson ctx "$num_ctx" \
        --arg ka "$BENCH_KEEP_ALIVE" \
        '{model:$model, prompt:("[" + $tag + "] " + $prompt), stream:true,
          keep_alive:$ka,
          options:{temperature:0, seed:42, num_predict:$np, num_ctx:$ctx}}' \
        >"$payload_file"

    start=$(date +%s%N)
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        if [ -z "$ttft_ms" ]; then
            now=$(date +%s%N)
            ttft_ms=$(((now - start) / 1000000))
        fi
        last=$line
    done < <(curl -sN "$OLLAMA_URL/api/generate" -d @"$payload_file")
    rm -f "$payload_file"

    [ -n "$last" ] || die "no response from $model (num_ctx=$num_ctx)"
    if err=$(jq -r '.error // empty' <<<"$last") && [ -n "$err" ]; then
        die "num_ctx=$num_ctx: $err"
    fi
    jq -r '"\(.eval_count // 0) \(.eval_duration // 0) \(.prompt_eval_count // 0) \(.prompt_eval_duration // 0) \(.load_duration // 0)"' \
        <<<"$last" | awk -v ttft="$ttft_ms" '{
            printf "%s %.1f %.1f %d %d %d\n",
                ttft,
                ($2 > 0) ? $1 / $2 * 1e9 : 0,
                ($4 > 0) ? $3 / $4 * 1e9 : 0,
                $5 / 1e6, $1, $3
        }'
}

# --- benchmark run ------------------------------------------------------

bench_ctx() {
    local model=$1 num_ctx=$2 csv=$3
    local metrics samples scenario prompt_file num_predict run prefill_tokens
    local tmp_prefill split reported

    log "=== $model num_ctx=$num_ctx: warm-up (cold load)"
    unload_all
    sampler_start
    if ! metrics=$(run_once "$model" "$BENCH_DIR/prompts/short.txt" 32 "$num_ctx" "warmup-$num_ctx"); then
        samples=$(sampler_stop)
        return 1
    fi
    samples=$(sampler_stop)
    split=$(model_gpu_split "$model")
    reported=$(model_reported_ctx "$model")
    log "num_ctx=$num_ctx loaded (api reports context_length=${reported:-?}, split=$split)"
    echo "$model,$num_ctx,warmup,0,$(tr ' ' , <<<"$metrics"),$(tr ' ' , <<<"$samples"),$split" >>"$csv"

    # generation: short instruction, fixed output length
    prompt_file=$BENCH_DIR/prompts/short.txt
    num_predict=$BENCH_NUM_PREDICT
    for run in $(seq 1 "$BENCH_RUNS"); do
        log "num_ctx=$num_ctx: generation run $run/$BENCH_RUNS"
        sampler_start
        metrics=$(run_once "$model" "$prompt_file" "$num_predict" "$num_ctx" "gen-$num_ctx-$run")
        samples=$(sampler_stop)
        split=$(model_gpu_split "$model")
        echo "$model,$num_ctx,generation,$run,$(tr ' ' , <<<"$metrics"),$(tr ' ' , <<<"$samples"),$split" >>"$csv"
    done

    # prefill: synthetic prompt ≈ 75% of the reserved context
    prefill_tokens=$(awk -v c="$num_ctx" -v f="$BENCH_PREFILL_FRAC" \
        'BEGIN { t = int(c * f); if (t < 256) t = 256; print t }')
    tmp_prefill=$(mktemp)
    make_prefill_prompt "$prefill_tokens" "$tmp_prefill" >/dev/null
    for run in $(seq 1 "$BENCH_RUNS"); do
        log "num_ctx=$num_ctx: prefill(~${prefill_tokens} tok target) run $run/$BENCH_RUNS"
        sampler_start
        if ! metrics=$(run_once "$model" "$tmp_prefill" 32 "$num_ctx" "prefill-$num_ctx-$run"); then
            sampler_stop >/dev/null || true
            rm -f "$tmp_prefill"
            return 1
        fi
        samples=$(sampler_stop)
        split=$(model_gpu_split "$model")
        echo "$model,$num_ctx,prefill,$run,$(tr ' ' , <<<"$metrics"),$(tr ' ' , <<<"$samples"),$split" >>"$csv"
    done
    rm -f "$tmp_prefill"
    return 0
}

cmd_run() {
    local stamp gpu_slug csv meta num_ctx native_ctx
    curl -sf "$OLLAMA_URL/api/version" >/dev/null || die "Ollama API not reachable at $OLLAMA_URL"
    curl -s "$OLLAMA_URL/api/tags" | jq -e --arg m "$BENCH_MODEL" \
        '.models[] | select(.name==$m)' >/dev/null \
        || die "model $BENCH_MODEL not present (ollama pull it first)"

    native_ctx=$(curl -s "$OLLAMA_URL/api/show" -d "$(jq -n --arg m "$BENCH_MODEL" '{name:$m}')" |
        jq -r '
          .model_info // {} | to_entries[]
          | select(.key|test("context_length$"))
          | .value' | head -1)
    native_ctx=${native_ctx:-unknown}

    stamp=$(date +%Y-%m-%d)
    gpu_slug=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 |
        tr '[:upper:] ' '[:lower:]-' | tr -cd 'a-z0-9-')
    mkdir -p "$BENCH_DIR/raw"
    csv=$BENCH_DIR/raw/ctx-$stamp-$gpu_slug.csv
    meta=${csv%.csv}.meta
    echo "$CSV_HEADER" >"$csv"

    {
        echo "date=$stamp"
        echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
        echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
        echo "cpu=$(lscpu | awk -F': +' '/Model name/ {print $2; exit}')"
        echo "ram=$(awk '/^MemTotal/ {printf "%d GB", $2/1024/1024}' /proc/meminfo)"
        echo "os=$(. /etc/os-release && echo "$PRETTY_NAME")"
        echo "ollama=$(curl -s "$OLLAMA_URL/api/version" | jq -r .version)"
        echo "model=$BENCH_MODEL"
        echo "native_ctx=$native_ctx"
        echo "ctx_sizes=$BENCH_CTX_SIZES"
        echo "runs=$BENCH_RUNS"
        echo "num_predict=$BENCH_NUM_PREDICT"
        echo "prefill_frac=$BENCH_PREFILL_FRAC"
    } >"$meta"

    log "stopping freki-vllm / freki-comfyui if present (clean VRAM)"
    docker stop freki-vllm freki-comfyui >/dev/null 2>&1 || true
    sleep 2

    for num_ctx in $BENCH_CTX_SIZES; do
        if ! bench_ctx "$BENCH_MODEL" "$num_ctx" "$csv"; then
            log "stopping sweep at num_ctx=$num_ctx (failed — typically VRAM / OOM)"
            echo "failed_at=$num_ctx" >>"$meta"
            break
        fi
        echo "ok_ctx=$num_ctx" >>"$meta"
    done
    unload_all

    cmd_report "$csv"
    log "done: $csv"
}

# --- report -------------------------------------------------------------

meta_get() { awk -F= -v k="$1" '$1==k {sub(/^[^=]*=/, ""); print; exit}' "$2"; }

cmd_report() {
    local csv=$1 meta=${1%.csv}.meta out=$BENCH_DIR/RESULTS-ctx.md
    [ -f "$csv" ] || die "no such file: $csv"
    [ -f "$meta" ] || die "missing metadata file: $meta"

    local agg
    agg=$(awk -F, '
        function median(list, n,  arr, i, tmp, j) {
            n = split(list, arr, " ")
            for (i = 2; i <= n; i++) {
                tmp = arr[i]
                for (j = i - 1; j >= 1 && arr[j] > tmp; j--) arr[j+1] = arr[j]
                arr[j+1] = tmp
            }
            return (n % 2) ? arr[(n+1)/2] : (arr[n/2] + arr[n/2+1]) / 2
        }
        NR == 1 { next }
        $3 == "warmup" {
            cold[$1 SUBSEP $2] = $8
            split_w[$1 SUBSEP $2] = $13
            if ($11 > vram_w[$1 SUBSEP $2]) vram_w[$1 SUBSEP $2] = $11
            next
        }
        {
            key = $1 SUBSEP $2 SUBSEP $3
            if (!(key in seen)) { seen[key] = 1; order[++n] = key }
            ttft[key] = ttft[key] " " $5
            gen[key] = gen[key] " " $6
            prompt[key] = prompt[key] " " $7
            if ($11 > vram[key]) vram[key] = $11
            if ($12 > ram[key]) ram[key] = $12
            split_m[key] = $13
            # track last prompt_eval for prefill size
            pe[key] = $10
        }
        END {
            for (i = 1; i <= n; i++) {
                key = order[i]
                split(key, k, SUBSEP)
                ck = k[1] SUBSEP k[2]
                # Pipe-separated: placement labels contain spaces ("100% GPU").
                printf "%s|%s|%s|%d|%.1f|%.1f|%d|%d|%d|%s|%d\n",
                    k[1], k[2], k[3],
                    median(ttft[key]), median(gen[key]), median(prompt[key]),
                    vram[key], ram[key], cold[ck],
                    (split_m[key] != "" ? split_m[key] : split_w[ck]),
                    pe[key]
            }
        }' "$csv")

    {
        echo "# Context-window results — \`$(meta_get model "$meta")\`"
        echo
        echo "> Auto-generated by [\`scripts/bench-ctx.sh\`](../scripts/bench-ctx.sh)" \
            "from [\`raw/$(basename "$csv")\`](raw/$(basename "$csv")) — do not edit by hand."
        echo
        echo "Measured on **$(meta_get date "$meta")**:"
        echo
        echo "- GPU: $(meta_get gpu "$meta") (driver $(meta_get driver "$meta"))"
        echo "- CPU: $(meta_get cpu "$meta") · RAM: $(meta_get ram "$meta")"
        echo "- OS: $(meta_get os "$meta") · Ollama $(meta_get ollama "$meta")"
        echo "- Model: \`$(meta_get model "$meta")\` (native context length **$(meta_get native_ctx "$meta")**)"
        echo "- Sweep: \`$(meta_get ctx_sizes "$meta")\` · prefill fill ≈ $(meta_get prefill_frac "$meta") × num_ctx"
        if failed=$(meta_get failed_at "$meta") && [ -n "$failed" ]; then
            echo "- Sweep stopped at \`num_ctx=$failed\` (OOM or API error)"
        fi
        echo
        echo "Ollama only **reserves** the KV cache for \`options.num_ctx\` (or the"
        echo "Modelfile \`PARAMETER num_ctx\`). The GGUF may allow 256k+, but the"
        echo "server default is much smaller — too small for long OpenCode sessions."
        echo
        echo "## Generation (short prompt, $(meta_get num_predict "$meta") tokens out)"
        echo
        echo "| num_ctx | Placement | VRAM peak | Cold load | TTFT | Generation |"
        echo "| ---: | --- | ---: | ---: | ---: | ---: |"
        awk -F'|' '$3 == "generation" {
                printf "| %s | %s | %s MiB | %.1f s | %s ms | **%.1f tok/s** |\n",
                    $2, $10, $7, $9 / 1000, $4, $5
            }' <<<"$agg"
        echo
        echo "## Prefill (synthetic prompt ≈ $(meta_get prefill_frac "$meta") × num_ctx, 32 tokens out)"
        echo
        echo "| num_ctx | Prompt tokens (median run) | TTFT | Prompt processing | Generation | VRAM peak |"
        echo "| ---: | ---: | ---: | ---: | ---: | ---: |"
        awk -F'|' '$3 == "prefill" {
                printf "| %s | %s | %s ms | **%.0f tok/s** | %.1f tok/s | %s MiB |\n",
                    $2, $11, $4, $6, $5, $7
            }' <<<"$agg"
        echo
        echo "## Using a larger context with OpenCode"
        echo
        echo "Per request (OpenAI-compatible or native), pass \`num_ctx\`:"
        echo
        echo '```bash'
        echo "# example: 32k context for a single generate call"
        echo "curl -s $OLLAMA_URL/api/generate -d '{"
        echo "  \"model\": \"$(meta_get model "$meta")\","
        echo "  \"prompt\": \"...\","
        echo "  \"options\": { \"num_ctx\": 32768 }"
        echo "}'"
        echo '```'
        echo
        echo "Persist it on the model so every client (including OpenCode) inherits it:"
        echo
        echo '```bash'
        echo "cat > /tmp/Modelfile.ornith-32k <<'EOF'"
        echo "FROM $(meta_get model "$meta")"
        echo "PARAMETER num_ctx 32768"
        echo "EOF"
        echo "ollama create ornith-32k -f /tmp/Modelfile.ornith-32k"
        echo "# then point OpenCode at model ornith-32k"
        echo '```'
        echo
        echo "Pick the largest \`num_ctx\` from the table that still fits your GPU with"
        echo "headroom for the desktop and any other process. Cold load grows with"
        echo "\`num_ctx\` because the KV cache is allocated up front."
        echo
        echo "## Method"
        echo
        echo "- Per \`num_ctx\`: unload all models, 1 warm-up (cold load with that"
        echo "  context), then $(meta_get runs "$meta") measured **generation** runs"
        echo "  (short prompt) and $(meta_get runs "$meta") **prefill** runs"
        echo "  (synthetic dump ≈ $(meta_get prefill_frac "$meta") × num_ctx tokens,"
        echo "  32 tokens out). Tables show medians."
        echo "- TTFT is client-side to the first streamed token; token rates from"
        echo "  Ollama \`eval_*\` / \`prompt_eval_*\` counters. \`temperature=0\`,"
        echo "  \`seed=42\`; unique prompt prefixes defeat the prompt cache."
        echo "- VRAM peak is total GPU memory (sampled every 0.5 s), including a"
        echo "  small desktop baseline."
    } >"$out"
    log "wrote $out"
}

case ${1:-run} in
run) cmd_run ;;
report) shift; cmd_report "${1:?usage: bench-ctx.sh report <csv>}" ;;
*) die "usage: bench-ctx.sh [run | report <csv>]" ;;
esac
