# Kubernetes manifests

Independent deployments for the same three stacks as `compose/`:

| Stack | Apply | Service (in-cluster) | GPU | PVC |
| --- | --- | --- | --- | --- |
| Ollama | `kubectl apply -k k8s/ollama` | `ollama.freki-llm:11434` | 1× `nvidia.com/gpu` | `ollama-models` 100 Gi |
| vLLM | `kubectl apply -k k8s/vllm` | `vllm.freki-llm:8000` | 1× `nvidia.com/gpu` | `vllm-huggingface` 100 Gi |
| ComfyUI | `kubectl apply -k k8s/comfyui` | `comfyui.freki-llm:8188` | 1× `nvidia.com/gpu` | models 80 Gi + output 20 Gi |

They share the `freki-llm` namespace but **not** the GPU: schedule only one
heavy workload per card (same rule as docker-compose on a single workstation).

## Prerequisites

- Kubernetes with a working **NVIDIA device plugin** (GPU Operator or
  [nvidia-device-plugin](https://github.com/NVIDIA/k8s-device-plugin))
- Nodes that advertise `nvidia.com/gpu`
- A default `StorageClass` that can bind `ReadWriteOnce` volumes (or set
  `storageClassName` on each PVC)
- `kubectl` with kustomize support (`kubectl apply -k`, kubectl ≥ 1.14)

Optional: if pods stay `Pending` with a RuntimeClass error, uncomment
`runtimeClassName: nvidia` in the Deployments (GPU Operator installs that class).

## Apply

```bash
# one stack at a time (recommended on a single GPU)
kubectl apply -k k8s/ollama

# wait until Ready
kubectl -n freki-llm rollout status deploy/ollama

# pull a model into the PVC (exec into the pod)
kubectl -n freki-llm exec -it deploy/ollama -- ollama pull ornith:9b

# smoke from your laptop
kubectl -n freki-llm port-forward svc/ollama 11434:11434
../../scripts/smoke-test.sh
```

vLLM (first start downloads weights into the PVC):

```bash
kubectl apply -k k8s/vllm
kubectl -n freki-llm rollout status deploy/vllm
kubectl -n freki-llm port-forward svc/vllm 8000:8000
../../scripts/smoke-test-vllm.sh
```

ComfyUI:

```bash
kubectl apply -k k8s/comfyui
kubectl -n freki-llm port-forward svc/comfyui 8188:8188
# then download checkpoints into the pod (adapt pull-image-models.sh paths)
```

## Change the vLLM model

Edit `args` in [`vllm/deployment.yaml`](vllm/deployment.yaml) (first argument
is the Hugging Face id). For gated models:

```bash
kubectl -n freki-llm create secret generic freki-hf-token \
  --from-literal=token="$HF_TOKEN"
```

The Deployment already mounts that secret as `HF_TOKEN` when present.

## Design notes

- **`strategy: Recreate`** — RWO model volumes cannot attach to two pods; rolling
  updates would deadlock.
- **`nvidia.com/gpu: 1`** on requests and limits — the device plugin schedules
  the pod onto a GPU node and injects devices.
- **GPU taint toleration** — common on managed GPU pools.
- **vLLM `/dev/shm`** — 8 Gi memory-backed `emptyDir` replaces compose `ipc: host`.
- **Probes** — Ollama `/api/version`, vLLM `/health`, ComfyUI `/`.
- Images and digests match `compose/` for reproducibility.

## Tear down

```bash
kubectl delete -k k8s/ollama
# PVCs are retained by default if you only delete the Deployment; to wipe weights:
kubectl -n freki-llm delete pvc ollama-models
```
