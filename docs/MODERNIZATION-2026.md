# Dream Textures Modernization Plan (2026)

Status: living document. Phase 1 is implemented in this fork; later phases are planned.
Research basis: multi-agent deep-research session (2026-07-13) with adversarially verified claims;
sources cited inline.

## Where the project stood (v0.4.1, Aug 2024)

- `diffusers==0.27.2`, `torch==2.3.1+cu118`, Python 3.10/3.11 release zips — predates *all* of
  diffusers' native quantization (bnb NF4 arrived in 0.31.0, GGUF/TorchAO in 0.32.0) and every
  post-SDXL pipeline (Flux 0.30, SD3.5 0.31, ControlNet Union 0.32, Chroma 0.34, Qwen-Image 0.35,
  Z-Image 0.36).
- `generator_process/` is hardwired to the SD U-Net architecture: `ModelType` is literally the
  U-Net `in_channels` (4/5/7/9), latent previews use SD1.5 RGB factors, sizing math assumes
  `pipe.unet` + `vae_scale_factor`, configs come from LDM YAMLs in `sd_configs/`.
- The `api/` layer (Backend, GenerationArguments, Task, GenerationResult) is genuinely
  model-agnostic and pluggable — the right foundation to keep.
- Release zips bundle deps compiled per Python version; Blender 5.1 (Python 3.13) cannot load the
  cp311 builds (the "bad magic number in 'transformers'" freeze).

## Phase 1 — Blender 5.1 / Python 3.13 port (DONE in this fork)

- New `requirements/win-linux-cuda-51.txt`: torch 2.13.0+cu130 (Turing+ GPUs), diffusers 0.39.0,
  transformers 5.x, huggingface_hub 1.x, plus peft/sentencepiece/protobuf/gguf for modern models.
- Ported removed diffusers APIs:
  - `convert_from_ckpt.download_from_original_stable_diffusion_ckpt` → `from_single_file`
    (`load_model.py`, `convert_original_stable_diffusion_to_diffusers.py`)
  - `original_config_file=` → `original_config=`
  - `diffusers.utils.hub_utils.old_diffusers_cache` → removed (legacy cache scan dropped)
  - `variant_compatible_siblings` private API → try/retry on `DiffusionPipeline.download`
  - `MultiControlNetModel` import moved to `diffusers.models.controlnets.multicontrolnet`
  - `resume_download=` kwarg removed (gone in huggingface_hub 1.x)
- transformers 5.x: `DPTFeatureExtractor` → `DPTImageProcessor` (`depth_to_image.py`).
- Bugfixes:
  - `actor.py`: response-queue wait now polls with a 1s timeout and raises a clear error when the
    backend process is dead (was: Blender froze forever).
  - `actor.py`: `__main__.__file__` cleared around `process.start()` for *any* file (was: only
    `"<blender string>"`), fixing crashes under `blender --python`.
  - `huggingface_hub.py`: tqdm progress hook guards `total=None` (new hub creates bars before the
    size is known — the `int / NoneType` crash on 4.1).
- Verified: full symbol smoke test passes under Blender's Python 3.13.9; addon registers in
  headless Blender 5.1.2; actor subprocess round-trips.

## Phase 2 — Modern model support (in the diffusers backend)

Target models, chosen for texture work on consumer VRAM (this dev machine: RTX 3070 8 GB):

| Model | Params / license | diffusers | 8 GB VRAM path | Notes |
|---|---|---|---|---|
| SDXL + ControlNet Union | 3.5B, OpenRAIL | AutoPipeline (Union integrated 0.32) | fp16 native | Baseline; one ControlNet for all conditions |
| Z-Image-Turbo | 6B S3-DiT, Apache 2.0 | `ZImagePipeline` (0.36+) | GGUF Q4/Q5 transformer + offloaded Qwen3-4B text encoder | 8 steps, cfg 0; best default candidate (16 GB BF16 unquantized) |
| Flux.1-schnell / Chroma (8.9B, Apache 2.0) | 12B / 8.9B | Flux pipelines (0.30+), Chroma (0.34+) | GGUF Q2_K–Q4_K + `enable_model_cpu_offload` | Flux Fill = inpaint/outpaint; Flux Control = depth/canny |
| Qwen-Image | 20B MoE-ish, Apache 2.0 | 0.35+ | GGUF low-quant + offload; tight on 8 GB | Strong text rendering; better on 16 GB+ cards |
| SD3.5 Medium/Large | 2.6B/8B, community license | 0.31+ | Medium fp16 / Large NF4 | Optional; Forge Neo dropped SD3 entirely |

Key mechanics (all verified against diffusers docs/releases):
- **GGUF**: per-component only — load the transformer via
  `Model.from_single_file(gguf_path, quantization_config=GGUFQuantizationConfig(compute_dtype=...))`,
  then assemble the pipeline. 11 quant types; weights dequantize per-forward. Requires `gguf` pip
  package (already in Phase 1 requirements).
- **On-the-fly quant**: `PipelineQuantizationConfig` (0.34+) with per-component `quant_mapping`
  (bitsandbytes 4/8-bit first choice on NVIDIA); FP8 layerwise casting via
  `enable_layerwise_casting()`.
- **Offload ladders**: `enable_model_cpu_offload` → group offload → group offload with
  `offload_to_disk_path` (0.34+) for low-RAM machines.

Code changes required:
1. Generalize `ModelType`/model detection — stop keying on U-Net `in_channels`; read
   `model_index.json` `_class_name` and transformer configs (DiT models have no `unet`).
2. Replace `pipe.unet` assumptions (`_configure_model_padding`, sizing, optimizations) with
   component discovery (`getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)`).
3. Per-architecture latent preview factors (current fast preview is SD1.5-only); flow-matching
   models need different step preview handling.
4. Scheduler surface: flow-matching schedulers (FlowMatchEulerDiscrete etc.) for
   Flux/SD3/Z-Image; hide UNet-era schedulers for those models.
5. Seamless-axes patching (`asymmetric padding on Conv2d`) needs a DiT equivalent — research says
   circular padding applies to the VAE conv layers, which still exist on all these models.
6. UI: quantization dropdown (auto/NF4/Q8/Q4/fp16), model-family-aware defaults (steps/cfg),
   GGUF single-file support in model linking.

## Phase 3 — ComfyUI backend (out-of-process)

The `api/Backend` interface already supports third-party backends (subclass discovery). Add a
built-in `ComfyUIBackend`:
- Talks to a running ComfyUI server over HTTP/WebSocket (`/prompt`, `/ws` progress, image fetch).
- Maps `GenerationArguments`/`Task` onto workflow templates (txt2img, img2img, inpaint, depth
  ControlNet, upscale) with per-model workflow JSON.
- Gets ComfyUI's memory management for free — their "Dynamic VRAM" allocator (comfy-aimdo,
  Mar 2026) fault-pages weights just-in-time and is far beyond anything we should reimplement.
- Prior art to borrow from: StableGen (sakalond/StableGen — SDXL/Flux/Qwen texturing via ComfyUI),
  AIGODLIKE/ComfyUI-BlenderAI-node, alexisrolland/ComfyUI-Blender.

This gives users with existing ComfyUI installs (and AMD users — ComfyUI has official ROCm-on-
Windows support since Jan 2026) a zero-duplication path, while the diffusers backend remains the
batteries-included default.

## Phase 4 — Packaging & platform

- **Blender 4.2+ extensions platform**: `blender_manifest.toml` + bundled wheels is the official
  path, but multi-GB torch wheels don't fit the extension model. Plan: stay a legacy addon
  short-term (still supported in 5.1); revisit an extension shell that installs the heavy deps
  into a user-writable location on first run.
- **CI**: add Python 3.13 (Blender 4.5/5.x) build matrix leg (`-5-1` suffix); drop the
  bytecode-only transformers zip (`zip_dependencies.py`) — it breaks on any Python mismatch and
  Windows long-path support makes it unnecessary; add a Linux build leg (currently absent).
- **Windows AMD**: drop DirectML patches. AMD ships native Windows ROCm wheels
  (torch 2.9.1+rocm7.2.1, Python 3.12) — incompatible with Blender's 3.13 today, so AMD-on-
  Windows users should use the ComfyUI backend (official ROCm Windows support) until ROCm wheels
  catch up to 3.13.
- **macOS**: torch MPS current wheels; needs its own requirements refresh and testing.

## Open questions

- Texture-specific quality (tileability, PBR-friendliness) across Z-Image/Flux/Qwen/Chroma is
  unbenchmarked — worth a bake-off once Phase 2 lands.
- Whether to keep the custom render engine + node graph (`engine/`) or deprecate it in favor of
  the operator workflow + ComfyUI graphs; `nodeitems_utils` still works in 5.1 but is slated for
  removal.
- Blender extensions platform limits (network permission, wheel size) for a future extension build.
