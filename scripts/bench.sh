#!/usr/bin/env bash
# Benchmark harness for the Ollama stack.
#
#   bench.sh run             run the benchmark matrix, write CSV + RESULTS.md
#   bench.sh report <csv>    regenerate RESULTS.md from an existing CSV
#
# Per model and scenario: 1 warm-up run (captures cold-load time), then
# BENCH_RUNS measured runs; the report shows the median. Time-to-first-token
# is measured client-side on the streaming API; token rates come from the
# counters Ollama returns with each response. VRAM and host RAM are sampled
# every 0.5 s while a request is in flight.
#
# Requires: curl, jq, awk, nvidia-smi.
set -euo pipefail
export LC_ALL=C # decimal points, not locale commas — the CSV depends on it

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_DIR=$(dirname "$SCRIPT_DIR")/benchmarks

OLLAMA_URL=${OLLAMA_URL:-http://localhost:11434}
BENCH_MODELS=${BENCH_MODELS:-"qwen3.5:4b mistral:7b llama3.1:8b ornith:9b qwen3.5:9b qwen3.5:9b-q8_0 gemma3:12b qwen3.6:35b"}
BENCH_RUNS=${BENCH_RUNS:-3}
BENCH_NUM_PREDICT=${BENCH_NUM_PREDICT:-256}
BENCH_KEEP_ALIVE=${BENCH_KEEP_ALIVE:-15m}

CSV_HEADER="model,scenario,run,ttft_ms,gen_tps,prompt_tps,load_ms,eval_count,prompt_eval_count,vram_peak_mib,ram_used_peak_mib"

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

for cmd in curl jq awk nvidia-smi; do
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

# Prints "<peak vram MiB> <peak host RAM used MiB>" (RAM = total - available).
sampler_stop() {
    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
    awk -v total="$(awk '/^MemTotal/ {print int($2/1024)}' /proc/meminfo)" '
        { if ($1 > v) v = $1; used = total - $2; if (used > r) r = used }
        END { print v+0, r+0 }' "$SAMPLER_FILE"
    rm -f "$SAMPLER_FILE"
}

cleanup() { [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# --- ollama helpers -----------------------------------------------------

unload_all() {
    curl -s "$OLLAMA_URL/api/ps" | jq -r '.models[].name' | while read -r m; do
        curl -s "$OLLAMA_URL/api/generate" \
            -d "$(jq -n --arg m "$m" '{model:$m, keep_alive:0}')" >/dev/null
    done
    sleep 3
}

model_disk_gb() {
    curl -s "$OLLAMA_URL/api/tags" |
        jq -r --arg m "$1" '.models[] | select(.name==$m) | .size / 1e9 | .*10 | round / 10'
}

# "100% GPU" or "72% GPU / 28% CPU", from how much of the loaded model sits in VRAM.
model_gpu_split() {
    curl -s "$OLLAMA_URL/api/ps" | jq -r --arg m "$1" '
        .models[] | select(.name==$m) |
        (.size_vram / .size * 100 | round) as $g |
        if $g >= 100 then "100% GPU" else "\($g)% GPU / \(100-$g)% CPU" end'
}

# --- one measured request -----------------------------------------------
# Prints: ttft_ms gen_tps prompt_tps load_ms eval_count prompt_eval_count
# The run tag is prepended to the prompt so Ollama's prompt cache cannot
# reuse the previous run's prefill.

run_once() {
    local model=$1 prompt_file=$2 num_predict=$3 tag=$4
    local payload start now ttft_ms="" last="" line

    payload=$(jq -n --arg model "$model" --arg tag "$tag" \
        --rawfile prompt "$prompt_file" \
        --argjson np "$num_predict" --arg ka "$BENCH_KEEP_ALIVE" \
        '{model:$model, prompt:("[" + $tag + "] " + $prompt), stream:true,
          keep_alive:$ka, options:{temperature:0, seed:42, num_predict:$np}}')

    start=$(date +%s%N)
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        if [ -z "$ttft_ms" ]; then
            now=$(date +%s%N)
            ttft_ms=$(((now - start) / 1000000))
        fi
        last=$line
    done < <(curl -sN "$OLLAMA_URL/api/generate" -d "$payload")

    [ -n "$last" ] || die "no response from $model"
    if err=$(jq -r '.error // empty' <<<"$last") && [ -n "$err" ]; then
        die "$model: $err"
    fi
    jq -r '"\(.eval_count) \(.eval_duration) \(.prompt_eval_count) \(.prompt_eval_duration) \(.load_duration)"' \
        <<<"$last" | awk -v ttft="$ttft_ms" '{
            printf "%s %.1f %.1f %d %d %d\n",
                ttft,
                ($2 > 0) ? $1 / $2 * 1e9 : 0,
                ($4 > 0) ? $3 / $4 * 1e9 : 0,
                $5 / 1e6, $1, $3
        }'
}

# --- benchmark run ------------------------------------------------------

bench_model() {
    local model=$1 csv=$2
    local metrics samples scenario prompt_file num_predict run

    log "=== $model: warm-up (cold load)"
    sampler_start
    metrics=$(run_once "$model" "$BENCH_DIR/prompts/short.txt" 32 "warmup")
    samples=$(sampler_stop)
    echo "$model,warmup,0,$(tr ' ' , <<<"$metrics"),$(tr ' ' , <<<"$samples")" >>"$csv"

    for scenario in generation long-prompt; do
        case $scenario in
        generation) prompt_file=$BENCH_DIR/prompts/short.txt num_predict=$BENCH_NUM_PREDICT ;;
        long-prompt) prompt_file=$BENCH_DIR/prompts/long.txt num_predict=128 ;;
        esac
        for run in $(seq 1 "$BENCH_RUNS"); do
            log "$model: $scenario run $run/$BENCH_RUNS"
            sampler_start
            metrics=$(run_once "$model" "$prompt_file" "$num_predict" "run $run")
            samples=$(sampler_stop)
            echo "$model,$scenario,$run,$(tr ' ' , <<<"$metrics"),$(tr ' ' , <<<"$samples")" >>"$csv"
        done
    done

    log "$model: split=$(model_gpu_split "$model")"
}

cmd_run() {
    local stamp gpu_slug csv meta model
    curl -sf "$OLLAMA_URL/api/version" >/dev/null || die "Ollama API not reachable at $OLLAMA_URL"

    stamp=$(date +%Y-%m-%d)
    gpu_slug=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 |
        tr '[:upper:] ' '[:lower:]-' | tr -cd 'a-z0-9-')
    mkdir -p "$BENCH_DIR/raw"
    csv=$BENCH_DIR/raw/$stamp-$gpu_slug.csv
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
        echo "runs=$BENCH_RUNS"
        echo "num_predict=$BENCH_NUM_PREDICT"
        for model in $BENCH_MODELS; do
            echo "disk_$model=$(model_disk_gb "$model")"
        done
    } >"$meta"

    for model in $BENCH_MODELS; do
        curl -s "$OLLAMA_URL/api/tags" | jq -e --arg m "$model" \
            '.models[] | select(.name==$m)' >/dev/null || die "model $model not present (ollama pull it first)"
    done

    for model in $BENCH_MODELS; do
        unload_all # cold-load measurement needs a clean slate
        bench_model "$model" "$csv"
        echo "split_$model=$(model_gpu_split "$model")" >>"$meta"
    done
    unload_all

    cmd_report "$csv"
    log "done: $csv"
}

# --- report -------------------------------------------------------------

meta_get() { awk -F= -v k="$1" '$1==k {sub(/^[^=]*=/, ""); print; exit}' "$2"; }

cmd_report() {
    local csv=$1 meta=${1%.csv}.meta out=$BENCH_DIR/RESULTS.md
    [ -f "$csv" ] || die "no such file: $csv"
    [ -f "$meta" ] || die "missing metadata file: $meta"

    # model scenario ttft_med gen_med prompt_med vram_max ram_max cold_load_ms
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
        $2 == "warmup" { cold[$1] = $7; next }
        {
            key = $1 SUBSEP $2
            if (!(key in seen)) { seen[key] = 1; order[++n] = key }
            ttft[key] = ttft[key] " " $4
            gen[key] = gen[key] " " $5
            prompt[key] = prompt[key] " " $6
            if ($10 > vram[key]) vram[key] = $10
            if ($11 > ram[key]) ram[key] = $11
        }
        END {
            for (i = 1; i <= n; i++) {
                key = order[i]
                split(key, k, SUBSEP)
                printf "%s %s %d %.1f %.1f %d %d %d\n", k[1], k[2],
                    median(ttft[key]), median(gen[key]), median(prompt[key]),
                    vram[key], ram[key], cold[k[1]]
            }
        }' "$csv")

    {
        echo "# Benchmark results"
        echo
        echo "> Auto-generated by [\`scripts/bench.sh\`](../scripts/bench.sh)" \
            "from [\`raw/$(basename "$csv")\`](raw/$(basename "$csv")) — do not edit by hand."
        echo
        echo "Measured on **$(meta_get date "$meta")**:"
        echo
        echo "- GPU: $(meta_get gpu "$meta") (driver $(meta_get driver "$meta"))"
        echo "- CPU: $(meta_get cpu "$meta") · RAM: $(meta_get ram "$meta")"
        echo "- OS: $(meta_get os "$meta") · Ollama $(meta_get ollama "$meta")"
        echo
        echo "## Generation (short prompt, $(meta_get num_predict "$meta") tokens out)"
        echo
        echo "| Model | Disk | Placement | VRAM peak | Cold load | TTFT | Generation |"
        echo "| --- | --- | --- | --- | --- | --- | --- |"
        awk -v meta="$meta" '
            BEGIN { while ((getline line < meta) > 0) { i = index(line, "="); m[substr(line, 1, i-1)] = substr(line, i+1) } }
            $2 == "generation" {
                printf "| `%s` | %s GB | %s | %s MiB | %.1f s | %s ms | **%.1f tok/s** |\n",
                    $1, m["disk_" $1], m["split_" $1], $6, $8 / 1000, $3, $4
            }' <<<"$agg"
        echo
        echo "## Long prompt (~1,200-token report, 128 tokens out)"
        echo
        echo "| Model | TTFT | Prompt processing | Generation |"
        echo "| --- | --- | --- | --- |"
        awk '$2 == "long-prompt" {
                printf "| `%s` | %s ms | %.0f tok/s | %.1f tok/s |\n", $1, $3, $5, $4
            }' <<<"$agg"
        echo
        echo "## Output quality"
        echo
        echo "Quality is a property of the model, not of this stack, so no scores are"
        echo "published here — instead, [unedited sample outputs](outputs/) of every"
        echo "benchmarked model on four fixed tasks (summarization, structured"
        echo "extraction, coding, arithmetic reasoning) are provided for side-by-side"
        echo "comparison. Generate them with \`scripts/sample-outputs.sh\`."
        echo
        echo "## Method"
        echo
        echo "- Per model and scenario: 1 discarded warm-up (its load time is the"
        echo "  \"cold load\" column), then $(meta_get runs "$meta") measured runs; tables show the **median**."
        echo "- TTFT (time to first token) is wall-clock from sending the request to the"
        echo "  first streamed token, measured client-side on the same host — it includes"
        echo "  prompt processing but no network latency."
        echo "- Token rates come from the \`eval_count\`/\`eval_duration\` counters returned"
        echo "  by the Ollama API. \`temperature=0\`, \`seed=42\`; each run's prompt gets a"
        echo "  unique prefix so Ollama's prompt cache cannot skip prefill."
        echo "- VRAM peak is total GPU memory used (sampled every 0.5 s), which includes"
        echo "  a small desktop baseline. \"Placement\" is how Ollama split the model"
        echo "  between GPU and CPU memory."
    } >"$out"
    log "wrote $out"
}

case ${1:-run} in
run) cmd_run ;;
report) shift; cmd_report "${1:?usage: bench.sh report <csv>}" ;;
*) die "usage: bench.sh [run | report <csv>]" ;;
esac
