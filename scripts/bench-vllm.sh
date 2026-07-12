#!/usr/bin/env bash
# Benchmark harness for the vLLM stack — apples-to-apples with scripts/bench.sh.
#
#   bench-vllm.sh run             run the matrix, write CSV + RESULTS-vllm.md
#   bench-vllm.sh report <csv>    regenerate RESULTS-vllm.md from an existing CSV
#
# Same scenarios and client-side metrics as the Ollama harness (TTFT, generation
# and prompt-processing rates, peak VRAM/RAM), against /v1/completions.
# Models are the Hugging Face counterparts of the Ollama RESULTS.md matrix
# (same family + ~same bit-width). See benchmarks/MODEL-MAP-vllm.md.
#
# vLLM serves one model per process: each pair recreates the compose service.
# Cold load is wall-clock from `docker compose up` to /health.
#
# Requires: curl, jq, awk, nvidia-smi, docker compose.
set -euo pipefail
export LC_ALL=C

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
BENCH_DIR=$ROOT_DIR/benchmarks
COMPOSE_DIR=$ROOT_DIR/compose/vllm

VLLM_URL=${VLLM_URL:-http://localhost:8000}

# ollama_tag=hf_id  — default = 16 GB-feasible rows of MODEL-MAP-vllm.md.
# Excluded on this class of card: qwen3.5:9b-q8_0 (FP8 multimodal OOM),
# gemma3:12b (tight), qwen3.6:35b (CPU-offload only under Ollama).
BENCH_PAIRS=${BENCH_PAIRS:-"\
qwen3.5:4b=cyankiwi/Qwen3.5-4B-AWQ-INT8-INT4 \
mistral:7b=solidrust/Mistral-7B-Instruct-v0.3-AWQ \
llama3.1:8b=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
ornith:9b=cyankiwi/Ornith-1.0-9B-AWQ-INT4 \
qwen3.5:9b=sanskar003/Qwen3.5-9B-AWQ"}

BENCH_RUNS=${BENCH_RUNS:-3}
BENCH_NUM_PREDICT=${BENCH_NUM_PREDICT:-256}
READY_TIMEOUT=${READY_TIMEOUT:-1200}
VLLM_IMAGE_TAG=${VLLM_IMAGE_TAG:-v0.25.0}
# 4096 covers the long-prompt scenario; keeps KV cache within 16 GB + desktop.
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.80}
VLLM_DTYPE=${VLLM_DTYPE:-auto}

# CSV uses the HF id as model; ollama counterpart lives in the .meta file.
CSV_HEADER="model,scenario,run,ttft_ms,gen_tps,prompt_tps,load_ms,eval_count,prompt_eval_count,vram_peak_mib,ram_used_peak_mib"

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

for cmd in curl jq awk nvidia-smi docker; do
    command -v "$cmd" >/dev/null || die "$cmd is required"
done
docker compose version >/dev/null 2>&1 || die "docker compose (v2) is required"

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

# --- compose helpers ----------------------------------------------------

compose() {
    (cd "$COMPOSE_DIR" && \
        VLLM_IMAGE_TAG="$VLLM_IMAGE_TAG" \
        VLLM_MODEL="$VLLM_MODEL" \
        VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
        VLLM_GPU_MEMORY_UTILIZATION="$VLLM_GPU_MEMORY_UTILIZATION" \
        VLLM_DTYPE="$VLLM_DTYPE" \
        HF_TOKEN="${HF_TOKEN:-}" \
        docker compose "$@")
}

wait_ready() {
    local deadline=$((SECONDS + READY_TIMEOUT))
    while (( SECONDS < deadline )); do
        if curl -sf "$VLLM_URL/health" >/dev/null 2>&1; then
            return 0
        fi
        # Surface fatal exits early
        if ! docker inspect -f '{{.State.Running}}' freki-vllm 2>/dev/null | grep -q true; then
            log "container freki-vllm is not running"
            docker logs --tail 60 freki-vllm >&2 || true
            return 1
        fi
        sleep 2
    done
    return 1
}

# Recreate the service with a new model. Prints cold-load time in ms to stdout.
start_model() {
    local model=$1
    local start now
    VLLM_MODEL=$model
    log "starting vLLM with $model"
    compose up -d --force-recreate >/dev/null
    start=$(date +%s%N)
    wait_ready || {
        log "container logs (tail):"
        docker logs --tail 100 freki-vllm >&2 || true
        die "vLLM did not become healthy for $model within ${READY_TIMEOUT}s"
    }
    now=$(date +%s%N)
    echo $(((now - start) / 1000000))
}

stop_vllm() {
    compose down >/dev/null 2>&1 || true
}

served_model() {
    curl -sf "$VLLM_URL/v1/models" | jq -r '.data[0].id // empty'
}

# Approximate on-disk size of a cached HF model repo (GB, one decimal).
model_disk_gb() {
    local model=$1 slug size
    slug=${model//\//--}
    size=$(docker run --rm --volumes-from freki-vllm alpine \
        sh -c "du -sb /root/.cache/huggingface/hub/models--${slug} 2>/dev/null | awk '{printf \"%.1f\", \$1/1e9}'" \
        2>/dev/null || true)
    if [[ -n "$size" && "$size" != "0.0" ]]; then
        echo "$size"
        return 0
    fi
    curl -sf "https://huggingface.co/api/models/${model}" |
        jq -r '((.usedStorage // 0) / 1e9 * 10 | round / 10)' 2>/dev/null || echo "?"
}

# --- one measured request -----------------------------------------------
# Prints: ttft_ms gen_tps prompt_tps load_ms eval_count prompt_eval_count
# load_ms is 0 for in-process requests (weights already resident).
# Token rates are client-side from stream usage (same wall-clock method for all).

run_once() {
    local model=$1 prompt_file=$2 num_predict=$3 tag=$4
    local payload start now ttft_ms="" first_ns="" end_ns usage_line="" line data
    local prompt_tokens=0 completion_tokens=0

    payload=$(jq -n --arg model "$model" --arg tag "$tag" \
        --rawfile prompt "$prompt_file" \
        --argjson np "$num_predict" \
        '{model:$model,
          prompt:("[" + $tag + "] " + $prompt),
          max_tokens:$np,
          temperature:0,
          seed:42,
          stream:true,
          stream_options:{include_usage:true}}')

    start=$(date +%s%N)
    while IFS= read -r line; do
        [[ "$line" == data:* ]] || continue
        data=${line#data: }
        data=${data# }
        [[ "$data" == "[DONE]" ]] && continue
        [[ -z "$data" ]] && continue
        if [[ -z "$ttft_ms" ]]; then
            if jq -e '(.choices[0].text // "") != ""' <<<"$data" >/dev/null 2>&1; then
                now=$(date +%s%N)
                first_ns=$now
                ttft_ms=$(((now - start) / 1000000))
            fi
        fi
        if jq -e '.usage != null' <<<"$data" >/dev/null 2>&1; then
            usage_line=$data
        fi
    done < <(curl -sN "$VLLM_URL/v1/completions" \
        -H "Content-Type: application/json" \
        -d "$payload")

    end_ns=$(date +%s%N)
    [[ -n "$usage_line" ]] || die "no usage in stream from $model (is stream_options supported?)"
    prompt_tokens=$(jq -r '.usage.prompt_tokens // 0' <<<"$usage_line")
    completion_tokens=$(jq -r '.usage.completion_tokens // 0' <<<"$usage_line")
    [[ -n "$ttft_ms" ]] || die "no streamed tokens from $model"
    [[ -n "$first_ns" ]] || first_ns=$((start + ttft_ms * 1000000))

    awk -v ttft="$ttft_ms" -v pt="$prompt_tokens" -v ct="$completion_tokens" \
        -v start="$start" -v first="$first_ns" -v end="$end_ns" 'BEGIN {
            gen_ns = end - first
            prefill_ns = first - start
            gen_tps = (gen_ns > 0 && ct > 0) ? ct / (gen_ns / 1e9) : 0
            prompt_tps = (prefill_ns > 0 && pt > 0) ? pt / (prefill_ns / 1e9) : 0
            printf "%s %.1f %.1f %d %d %d\n", ttft, gen_tps, prompt_tps, 0, ct, pt
        }'
}

# --- benchmark run ------------------------------------------------------

bench_model() {
    local model=$1 csv=$2 cold_ms=$3
    local metrics samples scenario prompt_file num_predict run

    log "=== $model: warm-up (already loaded; establishes steady state)"
    sampler_start
    metrics=$(run_once "$model" "$BENCH_DIR/prompts/short.txt" 32 "warmup")
    samples=$(sampler_stop)
    # cold_ms stored in load_ms of the warm-up row (mirrors Ollama load_duration)
    echo "$model,warmup,0,$(
        awk -v cold="$cold_ms" '{
            printf "%s,%s,%s,%s,%s,%s", $1, $2, $3, cold, $5, $6
        }' <<<"$metrics"
    ),$(tr ' ' , <<<"$samples")" >>"$csv"

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
}

cmd_run() {
    local stamp gpu_slug csv meta pair ollama_tag hf_id cold_ms disk served model_api
    local -a pairs

    # shellcheck disable=SC2206
    pairs=($BENCH_PAIRS)
    [[ ${#pairs[@]} -gt 0 ]] || die "BENCH_PAIRS is empty"

    stamp=$(date +%Y-%m-%d)
    gpu_slug=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 |
        tr '[:upper:] ' '[:lower:]-' | tr -cd 'a-z0-9-')
    mkdir -p "$BENCH_DIR/raw"
    csv=$BENCH_DIR/raw/vllm-$stamp-$gpu_slug.csv
    meta=${csv%.csv}.meta
    echo "$CSV_HEADER" >"$csv"

    {
        echo "date=$stamp"
        echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
        echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
        echo "cpu=$(lscpu | awk -F': +' '/Model name/ {print $2; exit}')"
        echo "ram=$(awk '/^MemTotal/ {printf "%d GB", $2/1024/1024}' /proc/meminfo)"
        echo "os=$(. /etc/os-release && echo "$PRETTY_NAME")"
        echo "vllm=$VLLM_IMAGE_TAG"
        echo "max_model_len=$VLLM_MAX_MODEL_LEN"
        echo "gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION"
        echo "dtype=$VLLM_DTYPE"
        echo "runs=$BENCH_RUNS"
        echo "num_predict=$BENCH_NUM_PREDICT"
        echo "map=benchmarks/MODEL-MAP-vllm.md"
    } >"$meta"

    log "stopping other freki GPU services for a clean VRAM baseline (ollama, comfyui)"
    docker stop freki-ollama freki-comfyui >/dev/null 2>&1 || true
    sleep 2

    for pair in "${pairs[@]}"; do
        ollama_tag=${pair%%=*}
        hf_id=${pair#*=}
        [[ -n "$ollama_tag" && -n "$hf_id" && "$pair" == *"="* ]] \
            || die "bad BENCH_PAIRS entry: $pair (want ollama_tag=hf_id)"

        cold_ms=$(start_model "$hf_id")
        served=$(served_model)
        model_api=$hf_id
        if [[ -n "$served" ]]; then
            model_api=$served
        fi
        if [[ "$model_api" != "$hf_id" ]]; then
            log "warning: requested $hf_id, server reports $model_api — using server id for requests"
        fi

        disk=$(model_disk_gb "$hf_id")
        {
            echo "disk_$model_api=$disk"
            echo "disk_$hf_id=$disk"
            echo "cold_$model_api=$cold_ms"
            echo "ollama_$model_api=$ollama_tag"
            echo "hf_$ollama_tag=$hf_id"
        } >>"$meta"
        log "$ollama_tag → $hf_id: cold load ${cold_ms} ms, disk ~ ${disk} GB"

        bench_model "$model_api" "$csv" "$cold_ms"
    done

    stop_vllm
    log "restarting freki-ollama / freki-comfyui if they were present"
    docker start freki-ollama freki-comfyui >/dev/null 2>&1 || true

    cmd_report "$csv"
    log "done: $csv"
}

# --- report -------------------------------------------------------------

meta_get() { awk -F= -v k="$1" '$1==k {sub(/^[^=]*=/, ""); print; exit}' "$2"; }

cmd_report() {
    local csv=$1 meta=${1%.csv}.meta out=$BENCH_DIR/RESULTS-vllm.md
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
        echo "# vLLM benchmark results"
        echo
        echo "> Auto-generated by [\`scripts/bench-vllm.sh\`](../scripts/bench-vllm.sh)" \
            "from [\`raw/$(basename "$csv")\`](raw/$(basename "$csv")) — do not edit by hand."
        echo
        echo "Measured on **$(meta_get date "$meta")**:"
        echo
        echo "- GPU: $(meta_get gpu "$meta") (driver $(meta_get driver "$meta"))"
        echo "- CPU: $(meta_get cpu "$meta") · RAM: $(meta_get ram "$meta")"
        echo "- OS: $(meta_get os "$meta") · vLLM image \`$(meta_get vllm "$meta")\`"
        echo "- \`max-model-len=$(meta_get max_model_len "$meta")\`," \
            "\`gpu-memory-utilization=$(meta_get gpu_memory_utilization "$meta")\`," \
            "\`dtype=$(meta_get dtype "$meta")\`"
        echo "- Model map: [\`MODEL-MAP-vllm.md\`](MODEL-MAP-vllm.md) (Ollama counterpart ↔ HF id)"
        echo
        echo "Compare with the Ollama numbers in [\`RESULTS.md\`](RESULTS.md) — same host,"
        echo "same prompts, same metrics. Quant schemes differ (GGUF Q4_K_M / Q8_0 vs"
        echo "AWQ / FP8) but each row is the same model family at a similar bit-width."
        echo
        echo "## Generation (short prompt, $(meta_get num_predict "$meta") tokens out)"
        echo
        echo "| Ollama counterpart | vLLM model | Disk | VRAM peak | Cold load | TTFT | Generation |"
        echo "| --- | --- | --- | --- | --- | --- | --- |"
        awk -v meta="$meta" '
            BEGIN {
                while ((getline line < meta) > 0) {
                    i = index(line, "=")
                    m[substr(line, 1, i-1)] = substr(line, i+1)
                }
            }
            $2 == "generation" {
                ollama = m["ollama_" $1]
                if (ollama == "") ollama = "?"
                disk = m["disk_" $1]
                if (disk == "") disk = "?"
                printf "| `%s` | `%s` | %s GB | %s MiB | %.1f s | %s ms | **%.1f tok/s** |\n",
                    ollama, $1, disk, $6, $8 / 1000, $3, $4
            }' <<<"$agg"
        echo
        echo "## Long prompt (~1,200-token report, 128 tokens out)"
        echo
        echo "| Ollama counterpart | vLLM model | TTFT | Prompt processing | Generation |"
        echo "| --- | --- | --- | --- | --- |"
        awk -v meta="$meta" '
            BEGIN {
                while ((getline line < meta) > 0) {
                    i = index(line, "=")
                    m[substr(line, 1, i-1)] = substr(line, i+1)
                }
            }
            $2 == "long-prompt" {
                ollama = m["ollama_" $1]
                if (ollama == "") ollama = "?"
                printf "| `%s` | `%s` | %s ms | %.0f tok/s | %.1f tok/s |\n",
                    ollama, $1, $3, $5, $4
            }' <<<"$agg"
        echo
        echo "## Method"
        echo
        echo "- Same prompts and scenarios as the [Ollama harness](RESULTS.md)."
        echo "- Per model: container recreated with that Hugging Face id, cold load"
        echo "  measured as wall-clock to \`/health\`, then 1 discarded warm-up and"
        echo "  $(meta_get runs "$meta") measured runs; tables show the **median**."
        echo "- TTFT is client-side wall-clock to the first streamed token on"
        echo "  \`/v1/completions\` (includes prefill, no network hop off-box)."
        echo "- Token rates are client-side from the final stream \`usage\` object:"
        echo "  prompt tok/s ≈ prompt_tokens / TTFT; generation tok/s ="
        echo "  completion_tokens / wall-time after the first token."
        echo "  \`temperature=0\`, \`seed=42\`; each run's prompt gets a unique prefix."
        echo "- VRAM peak is total GPU memory used (sampled every 0.5 s), including"
        echo "  a small desktop baseline **and** the KV-cache pool vLLM pre-allocates"
        echo "  up to \`gpu-memory-utilization\`. That is why peaks sit near the card"
        echo "  limit even for small models — unlike Ollama, which grows VRAM with"
        echo "  demand. There is no per-request load and no CPU offload split."
        echo "- Compose serves with \`--enforce-eager\` (no CUDA graphs) so multimodal"
        echo "  Qwen3.5 AWQ fits a 16 GB card alongside a desktop session; single-stream"
        echo "  rates are slightly lower than a fully graph-captured server."
        echo "- Rows omitted from the default matrix on this host (see model map):"
        echo "  \`qwen3.5:9b-q8_0\` (FP8 multimodal OOM), \`gemma3:12b\` (tight),"
        echo "  \`qwen3.6:35b\` (Ollama already CPU-offloads ~half)."
    } >"$out"
    log "wrote $out"
}

case ${1:-run} in
run) cmd_run ;;
report) shift; cmd_report "${1:?usage: bench-vllm.sh report <csv>}" ;;
*) die "usage: bench-vllm.sh [run | report <csv>]" ;;
esac
