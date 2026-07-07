#!/usr/bin/env bash
# Downloads the checkpoints used by the image-generation benchmark into the
# running ComfyUI container's model volume. All three are public downloads,
# no Hugging Face token needed. Requires `docker compose up -d` in
# compose/comfyui to already have been run.
#
# - SDXL Base 1.0: CreativeML Open RAIL++-M, commercially permissive.
# - FLUX.1-schnell fp8: Apache 2.0, commercially permissive.
# - FLUX.1-dev fp8: Black Forest Labs non-commercial license — the WEIGHTS
#   may not be run in a revenue-generating service without a paid license;
#   OUTPUT images are explicitly usable for any purpose including
#   commercial. See benchmarks/RESULTS-images.md for the exact wording.
set -euo pipefail

CONTAINER=${COMFYUI_CONTAINER:-freki-comfyui}
DEST=/root/ComfyUI/models/checkpoints

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

docker exec "$CONTAINER" true 2>/dev/null ||
    { echo "ERROR: container $CONTAINER not running — start it first with" \
           "'cd compose/comfyui && docker compose up -d'" >&2; exit 1; }

pull() {
    local name=$1 url=$2
    if docker exec "$CONTAINER" test -f "$DEST/$name"; then
        log "$name already present, skipping"
        return
    fi
    log "pulling $name"
    docker exec "$CONTAINER" bash -c \
        "mkdir -p '$DEST' && curl -fL --retry 3 -o '$DEST/$name.part' '$url' && mv '$DEST/$name.part' '$DEST/$name'"
    log "done: $name"
}

pull sd_xl_base_1.0.safetensors \
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
pull flux1-schnell-fp8.safetensors \
    "https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors"
pull flux1-dev-fp8.safetensors \
    "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors"

log "all image models present:"
docker exec "$CONTAINER" ls -lh "$DEST"
