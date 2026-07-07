#!/usr/bin/env bash
# Generates the image-generation quality gallery: one PNG per checkpoint x
# prompt under benchmarks/outputs/images/, plus an index markdown page for
# side-by-side comparison. No automated quality score, on the same
# rationale as scripts/sample-outputs.sh: image quality is a property of the
# model, not of this stack — judge with your own eyes.
#
# Requires: curl, jq.
set -euo pipefail
export LC_ALL=C

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_DIR=$(dirname "$SCRIPT_DIR")/benchmarks

COMFYUI_URL=${COMFYUI_URL:-http://localhost:8188}
WIDTH=${GALLERY_IMAGE_WIDTH:-1024}
HEIGHT=${GALLERY_IMAGE_HEIGHT:-1024}
PROMPTS=${GALLERY_IMAGE_PROMPTS:-"portrait product typography scene"}

# checkpoint:family:steps:cfg:guidance — same per-model defaults as bench-images.sh
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

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

for cmd in curl jq; do command -v "$cmd" >/dev/null || die "$cmd is required"; done
curl -sf "$COMFYUI_URL/system_stats" >/dev/null || die "ComfyUI API not reachable at $COMFYUI_URL"

build_graph() {
    local ckpt=$1 family=$2 steps=$3 cfg=$4 guidance=$5 seed=$6 prompt=$7
    if [ "$family" = flux ]; then
        jq -n --arg ckpt "$ckpt" --arg prompt "$prompt" \
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
          "8": {class_type:"SaveImage", inputs:{filename_prefix:"gallery", images:["7",0]}}
        }'
    else
        jq -n --arg ckpt "$ckpt" --arg prompt "$prompt" \
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
          "8": {class_type:"SaveImage", inputs:{filename_prefix:"gallery", images:["7",0]}}
        }'
    fi
}

generate() {
    local graph=$1 save_to=$2 prompt_id resp hist filename subfolder
    resp=$(curl -sf -X POST "$COMFYUI_URL/prompt" \
        -H 'Content-Type: application/json' \
        -d "$(jq -n --argjson g "$graph" '{prompt:$g, client_id:"freki-gallery"}')")
    prompt_id=$(jq -r '.prompt_id // empty' <<<"$resp")
    [ -n "$prompt_id" ] || die "submit failed: $(jq -c '.error // .' <<<"$resp")"

    while :; do
        hist=$(curl -sf "$COMFYUI_URL/history/$prompt_id")
        [ "$(jq -r --arg id "$prompt_id" '.[$id] // empty' <<<"$hist")" != "" ] && break
        sleep 0.5
    done

    if err=$(jq -r --arg id "$prompt_id" '.[$id].status.messages[]? | select(.[0]=="execution_error") | .[1].exception_message // empty' <<<"$hist") && [ -n "$err" ]; then
        die "generation failed: $err"
    fi

    filename=$(jq -r --arg id "$prompt_id" '.[$id].outputs | to_entries[0].value.images[0].filename' <<<"$hist")
    subfolder=$(jq -r --arg id "$prompt_id" '.[$id].outputs | to_entries[0].value.images[0].subfolder' <<<"$hist")
    curl -sf "$COMFYUI_URL/view?filename=$filename&subfolder=$subfolder&type=output" -o "$save_to" ||
        die "failed to fetch output image"
}

mkdir -p "$BENCH_DIR/outputs/images"

# Model-major order so each checkpoint is loaded into VRAM only once.
for spec in "${MODEL_SPECS[@]}"; do
    IFS=: read -r ckpt family steps cfg guidance <<<"$spec"
    docker exec freki-comfyui test -f "/root/ComfyUI/models/checkpoints/$ckpt" ||
        die "checkpoint $ckpt not present (run scripts/pull-image-models.sh first)"
    curl -s -X POST "$COMFYUI_URL/free" -H 'Content-Type: application/json' \
        -d '{"unload_models": true, "free_memory": true}' >/dev/null
    for p in $PROMPTS; do
        file="$BENCH_DIR/prompts/images/$p.txt"
        [ -f "$file" ] || die "missing prompt file: $file"
        prompt=$(cat "$file")
        out="$BENCH_DIR/outputs/images/${ckpt%.safetensors}_$p.png"
        log "$ckpt: $p"
        generate "$(build_graph "$ckpt" "$family" "$steps" "$cfg" "$guidance" 123 "$prompt")" "$out"
    done
done

{
    echo "# Image generation — sample gallery"
    echo
    echo "> Auto-generated by [\`scripts/image-gallery.sh\`](../../scripts/image-gallery.sh);"
    echo "> unedited outputs, seed 123, each checkpoint's own default steps/CFG (see"
    echo "> [RESULTS-images.md](../RESULTS-images.md))."
    echo
    for p in $PROMPTS; do
        echo "## $p"
        echo
        echo "> $(cat "$BENCH_DIR/prompts/images/$p.txt")"
        echo
        for spec in "${MODEL_SPECS[@]}"; do
            IFS=: read -r ckpt _ _ _ _ <<<"$spec"
            echo "**\`$ckpt\`**"
            echo
            echo "![${ckpt%.safetensors} $p](images/${ckpt%.safetensors}_$p.png)"
            echo
        done
    done
} >"$BENCH_DIR/outputs/images.md"
log "wrote $BENCH_DIR/outputs/images.md"
log "gallery done"
