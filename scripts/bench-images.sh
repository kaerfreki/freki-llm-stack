#!/usr/bin/env bash
# Performance harness for the ComfyUI image-generation stack.
#
#   bench-images.sh run             run the benchmark matrix, write CSV + RESULTS-images.md
#   bench-images.sh report <csv>    regenerate RESULTS-images.md from an existing CSV
#
# Per model: 1 discarded warm-up (captures cold checkpoint load), then
# BENCH_IMAGE_RUNS measured generations of one 1024x1024 image at each
# model's own commonly published default sampler settings (steps/cfg differ
# a lot between SDXL and the two FLUX variants, so pinning identical
# settings across all three would benchmark an artificial scenario rather
# than how each model is actually meant to be run — see the per-model
# settings below).
#
# Requires: curl, jq, awk, nvidia-smi.
set -euo pipefail
export LC_ALL=C

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_DIR=$(dirname "$SCRIPT_DIR")/benchmarks

COMFYUI_URL=${COMFYUI_URL:-http://localhost:8188}
BENCH_IMAGE_RUNS=${BENCH_IMAGE_RUNS:-3}
PROMPT=${BENCH_IMAGE_PROMPT:-"a weathered lighthouse on a rocky cliff at golden hour, dramatic clouds, photorealistic, highly detailed"}
WIDTH=${BENCH_IMAGE_WIDTH:-1024}
HEIGHT=${BENCH_IMAGE_HEIGHT:-1024}

CSV_HEADER="model,run,total_s,vram_peak_mib,ram_used_peak_mib"

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

for cmd in curl jq awk nvidia-smi; do
    command -v "$cmd" >/dev/null || die "$cmd is required"
done

# model:family:steps:cfg:guidance (guidance unused for sdxl family)
ALL_MODEL_SPECS=(
    "sd_xl_base_1.0.safetensors:sdxl:25:7.0:0"
    "flux1-schnell-fp8.safetensors:flux:4:1.0:0"
    "flux1-dev-fp8.safetensors:flux:50:1.0:3.5"
)
if [ -n "${BENCH_IMAGE_MODELS:-}" ]; then
    MODEL_SPECS=()
    for want in $BENCH_IMAGE_MODELS; do
        for spec in "${ALL_MODEL_SPECS[@]}"; do
            [[ $spec == "$want:"* ]] && MODEL_SPECS+=("$spec")
        done
    done
else
    MODEL_SPECS=("${ALL_MODEL_SPECS[@]}")
fi

# --- sampling (same pattern as scripts/bench.sh) --------------------------

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
}

cleanup() { [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT

unload_all() {
    curl -s -X POST "$COMFYUI_URL/free" \
        -H 'Content-Type: application/json' \
        -d '{"unload_models": true, "free_memory": true}' >/dev/null
    sleep 1
}

# --- graph construction ----------------------------------------------------
# Builds a ComfyUI API-format prompt graph. SDXL uses a plain
# CheckpointLoaderSimple; FLUX needs a FluxGuidance node between the positive
# CLIPTextEncode and the sampler, since FLUX uses guidance distillation
# rather than classic CFG (cfg is pinned to 1.0 for the FLUX fp8 checkpoints
# per Comfy-Org's own usage note).
build_graph() {
    local ckpt=$1 family=$2 steps=$3 cfg=$4 guidance=$5 seed=$6
    if [ "$family" = flux ]; then
        jq -n --arg ckpt "$ckpt" --arg prompt "$PROMPT" \
            --argjson steps "$steps" --argjson cfg "$cfg" --argjson guidance "$guidance" \
            --argjson seed "$seed" --argjson w "$WIDTH" --argjson h "$HEIGHT" '
        {
          "1": {class_type:"CheckpointLoaderSimple", inputs:{ckpt_name:$ckpt}},
          "2": {class_type:"CLIPTextEncode", inputs:{text:$prompt, clip:["1",1]}},
          "3": {class_type:"CLIPTextEncode", inputs:{text:"", clip:["1",1]}},
          "4": {class_type:"FluxGuidance", inputs:{guidance:$guidance, conditioning:["2",0]}},
          "5": {class_type:"EmptyLatentImage", inputs:{width:$w, height:$h, batch_size:1}},
          "6": {class_type:"KSampler", inputs:{
              seed:$seed, steps:$steps, cfg:$cfg, sampler_name:"euler", scheduler:"simple",
              denoise:1.0, model:["1",0], positive:["4",0], negative:["3",0], latent_image:["5",0]}},
          "7": {class_type:"VAEDecode", inputs:{samples:["6",0], vae:["1",2]}},
          "8": {class_type:"SaveImage", inputs:{filename_prefix:"bench", images:["7",0]}}
        }'
    else
        jq -n --arg ckpt "$ckpt" --arg prompt "$PROMPT" \
            --argjson steps "$steps" --argjson cfg "$cfg" \
            --argjson seed "$seed" --argjson w "$WIDTH" --argjson h "$HEIGHT" '
        {
          "1": {class_type:"CheckpointLoaderSimple", inputs:{ckpt_name:$ckpt}},
          "2": {class_type:"CLIPTextEncode", inputs:{text:$prompt, clip:["1",1]}},
          "3": {class_type:"CLIPTextEncode", inputs:{text:"", clip:["1",1]}},
          "5": {class_type:"EmptyLatentImage", inputs:{width:$w, height:$h, batch_size:1}},
          "6": {class_type:"KSampler", inputs:{
              seed:$seed, steps:$steps, cfg:$cfg, sampler_name:"euler", scheduler:"normal",
              denoise:1.0, model:["1",0], positive:["2",0], negative:["3",0], latent_image:["5",0]}},
          "7": {class_type:"VAEDecode", inputs:{samples:["6",0], vae:["1",2]}},
          "8": {class_type:"SaveImage", inputs:{filename_prefix:"bench", images:["7",0]}}
        }'
    fi
}

# Submits a graph, blocks until done, prints elapsed seconds. If save_to is
# given, also downloads the resulting PNG there.
run_graph() {
    local graph=$1 save_to=${2:-} start prompt_id resp elapsed hist filename subfolder

    start=$(date +%s%N)
    resp=$(curl -sf -X POST "$COMFYUI_URL/prompt" \
        -H 'Content-Type: application/json' \
        -d "$(jq -n --argjson g "$graph" '{prompt:$g, client_id:"freki-bench"}')")
    prompt_id=$(jq -r '.prompt_id // empty' <<<"$resp")
    [ -n "$prompt_id" ] || die "submit failed: $(jq -c '.error // .' <<<"$resp")"

    while :; do
        hist=$(curl -sf "$COMFYUI_URL/history/$prompt_id")
        [ "$(jq -r --arg id "$prompt_id" '.[$id] // empty' <<<"$hist")" != "" ] && break
        sleep 0.5
    done
    elapsed=$(awk -v ns="$(( $(date +%s%N) - start ))" 'BEGIN{printf "%.2f", ns/1e9}')

    if err=$(jq -r --arg id "$prompt_id" '.[$id].status.messages[]? | select(.[0]=="execution_error") | .[1].exception_message // empty' <<<"$hist") && [ -n "$err" ]; then
        die "generation failed: $err"
    fi

    if [ -n "$save_to" ]; then
        filename=$(jq -r --arg id "$prompt_id" '.[$id].outputs | to_entries[0].value.images[0].filename' <<<"$hist")
        subfolder=$(jq -r --arg id "$prompt_id" '.[$id].outputs | to_entries[0].value.images[0].subfolder' <<<"$hist")
        curl -sf "$COMFYUI_URL/view?filename=$filename&subfolder=$subfolder&type=output" -o "$save_to" ||
            die "failed to fetch output image"
    fi
    echo "$elapsed"
}

cmd_run() {
    local stamp gpu_slug csv meta spec ckpt family steps cfg guidance run elapsed samples
    curl -sf "$COMFYUI_URL/system_stats" >/dev/null || die "ComfyUI API not reachable at $COMFYUI_URL"

    stamp=$(date +%Y-%m-%d)
    gpu_slug=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 |
        tr '[:upper:] ' '[:lower:]-' | tr -cd 'a-z0-9-')
    mkdir -p "$BENCH_DIR/raw"
    csv=$BENCH_DIR/raw/images-$stamp-$gpu_slug.csv
    meta=${csv%.csv}.meta
    echo "$CSV_HEADER" >"$csv"

    {
        echo "date=$stamp"
        echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
        echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
        echo "comfyui=$(curl -s "$COMFYUI_URL/system_stats" | jq -r .system.comfyui_version)"
        echo "runs=$BENCH_IMAGE_RUNS"
        echo "resolution=${WIDTH}x${HEIGHT}"
        echo "prompt=$PROMPT"
    } >"$meta"

    for spec in "${MODEL_SPECS[@]}"; do
        IFS=: read -r ckpt family steps cfg guidance <<<"$spec"
        docker exec freki-comfyui test -f "/root/ComfyUI/models/checkpoints/$ckpt" ||
            die "checkpoint $ckpt not present (run scripts/pull-image-models.sh first)"
        echo "steps_$ckpt=$steps" >>"$meta"
        echo "cfg_$ckpt=$cfg" >>"$meta"

        unload_all
        log "=== $ckpt: warm-up (cold load)"
        run_graph "$(build_graph "$ckpt" "$family" "$steps" "$cfg" "$guidance" 42)" >/dev/null

        for run in $(seq 1 "$BENCH_IMAGE_RUNS"); do
            log "$ckpt: run $run/$BENCH_IMAGE_RUNS"
            sampler_start
            elapsed=$(run_graph "$(build_graph "$ckpt" "$family" "$steps" "$cfg" "$guidance" $((42 + run)))")
            samples=$(sampler_stop)
            echo "$ckpt,$run,$elapsed,$(tr ' ' , <<<"$samples")" >>"$csv"
        done
    done
    unload_all

    cmd_report "$csv"
    log "done: $csv"
}

meta_get() { awk -F= -v k="$1" '$1==k {sub(/^[^=]*=/, ""); print; exit}' "$2"; }

cmd_report() {
    local csv=$1 meta=${1%.csv}.meta out=$BENCH_DIR/RESULTS-images.md
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
        {
            if (!($1 in seen)) { seen[$1] = 1; order[++n] = $1 }
            t[$1] = t[$1] " " $3
            if ($4 > vram[$1]) vram[$1] = $4
            if ($5 > ram[$1]) ram[$1] = $5
        }
        END {
            for (i = 1; i <= n; i++) {
                m = order[i]
                s = median(t[m])
                printf "%s %.1f %.1f %d %d\n", m, s, 60/s, vram[m], ram[m]
            }
        }' "$csv")

    {
        echo "# Image generation results"
        echo
        echo "> Auto-generated by [\`scripts/bench-images.sh\`](../scripts/bench-images.sh)" \
            "from [\`raw/$(basename "$csv")\`](raw/$(basename "$csv")) — do not edit by hand."
        echo
        echo "Measured on **$(meta_get date "$meta")**, ComfyUI $(meta_get comfyui "$meta")," \
            "$(meta_get resolution "$meta"), median of $(meta_get runs "$meta") runs:"
        echo
        echo "- GPU: $(meta_get gpu "$meta") (driver $(meta_get driver "$meta"))"
        echo
        echo "| Checkpoint | Steps | CFG | Time / image | Images / min | VRAM peak | Host RAM peak |"
        echo "| --- | --- | --- | --- | --- | --- | --- |"
        awk -v meta="$meta" '
            BEGIN { while ((getline line < meta) > 0) { i = index(line, "="); m[substr(line, 1, i-1)] = substr(line, i+1) } }
            {
                printf "| `%s` | %s | %s | %.1f s | %.1f | %s MiB | %s MiB |\n",
                    $1, m["steps_" $1], m["cfg_" $1], $2, $3, $4, $5
            }' <<<"$agg"
        echo
        echo "## Method"
        echo
        echo "- Fixed prompt: \"$(meta_get prompt "$meta")\"."
        echo "- Per checkpoint: 1 discarded warm-up (cold checkpoint load), then" \
            "$(meta_get runs "$meta") measured generations; table shows the **median**."
        echo "- Steps/CFG are each checkpoint's own commonly published defaults, not a"
        echo "  single pinned setting — SDXL, FLUX.1-schnell and FLUX.1-dev are meant to"
        echo "  be run very differently, so forcing identical settings would benchmark"
        echo "  an artificial scenario rather than real usage. FLUX uses guidance"
        echo "  distillation (a \`FluxGuidance\` node) instead of classic CFG, which is"
        echo "  why its CFG is pinned to 1.0."
        echo "- VRAM/RAM peaks are sampled every 0.5 s during generation; models are"
        echo "  explicitly unloaded (\`POST /free\`) between checkpoints so cold-load and"
        echo "  peak-memory numbers aren't polluted by a previous model still resident."
    } >"$out"
    log "wrote $out"
}

case ${1:-run} in
run) cmd_run ;;
report) shift; cmd_report "${1:?usage: bench-images.sh report <csv>}" ;;
*) die "usage: bench-images.sh [run | report <csv>]" ;;
esac
