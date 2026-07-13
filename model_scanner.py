"""Discovers checkpoint files in user-registered model folders (e.g. a ComfyUI models directory).

Classification reads only the safetensors JSON header (a few KB) — weights are never loaded,
so scanning hundreds of files is fast enough to run on the main thread.
"""
import json
import os
from dataclasses import dataclass

from .generator_process.models import ModelType

CHECKPOINT_EXTENSIONS = {'.safetensors', '.sft', '.ckpt', '.pth'}

_MAX_HEADER_SIZE = 256 * 1024 * 1024


@dataclass
class ScannedModel:
    path: str
    root: str
    display: str
    model_type: ModelType


def _read_safetensors_header(path):
    try:
        with open(path, 'rb') as f:
            length = int.from_bytes(f.read(8), 'little')
            if length <= 0 or length > _MAX_HEADER_SIZE:
                return None
            return json.loads(f.read(length))
    except (OSError, ValueError):
        return None


def classify_checkpoint(path) -> ModelType | None:
    """Classify a checkpoint file, or return None if it isn't a usable base model
    (LoRA, VAE, embedding, unreadable, or an architecture the backend can't load yet)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.ckpt', '.pth'):
        # pickle checkpoints can't be inspected cheaply; let single-file loading auto-detect
        return ModelType.UNSPECIFIED_CHECKPOINT

    header = _read_safetensors_header(path)
    if header is None:
        return None
    keys = [k for k in header.keys() if k != '__metadata__']
    if not keys:
        return None

    def any_key(*subs):
        return any(any(s in k for s in subs) for k in keys)

    # non-checkpoint files
    if any_key('lora_down', 'lora_up', '.lora_A.', '.lora_B.', 'lora_te', '.hada_w', '.lokr_w', 'lora_transformer_'):
        return None  # LoRA
    if any_key('string_to_param', 'emb_params'):
        return None  # textual inversion embedding
    if all(k.startswith(('encoder.', 'decoder.', 'quant_conv', 'post_quant_conv', 'vae.')) for k in keys):
        return None  # standalone VAE
    if any(k.startswith('control_model.') for k in keys) or any_key('controlnet_cond_embedding'):
        return ModelType.CONTROL_NET
    # DiT-family single-file checkpoints (Flux, SD3, Z-Image, ...) — not loadable by the
    # current UNet-based backend; skip until DiT support lands
    if any_key('double_blocks.', 'single_blocks.', 'joint_blocks.', 'transformer_blocks.0.attn.add_q_proj'):
        return None

    # LDM-style UNet checkpoints (SD1.x / SD2.x / SDXL): in_channels distinguishes the task
    unet_in = header.get('model.diffusion_model.input_blocks.0.0.weight')
    if unet_in is not None:
        shape = unet_in.get('shape', [])
        in_channels = shape[1] if len(shape) > 1 else 4
        model_type = ModelType(in_channels)
        return ModelType.UNSPECIFIED_CHECKPOINT if model_type == ModelType.UNKNOWN else model_type
    if any(k.startswith(('conditioner.', 'cond_stage_model.')) for k in keys):
        return ModelType.UNSPECIFIED_CHECKPOINT

    return None


def infer_single_file_pipeline_class(path) -> str:
    """Best-effort diffusers pipeline class name for a single-file checkpoint, from its header.

    Used when a checkpoint is loaded with ModelConfig.AUTO_DETECT and the requested class
    (e.g. an AutoPipeline) can't load single files itself.
    """
    header = _read_safetensors_header(path) if path.lower().endswith(('.safetensors', '.sft')) else None
    if not header:
        return 'StableDiffusionPipeline'
    keys = [k for k in header.keys() if k != '__metadata__']
    is_xl = any(k.startswith('conditioner.embedders.') for k in keys)
    is_xl_refiner = is_xl and not any(k.startswith('conditioner.embedders.1.') for k in keys)
    unet_in = header.get('model.diffusion_model.input_blocks.0.0.weight', {})
    shape = unet_in.get('shape', [])
    in_channels = shape[1] if len(shape) > 1 else 4
    if is_xl:
        if is_xl_refiner:
            return 'StableDiffusionXLImg2ImgPipeline'
        return 'StableDiffusionXLInpaintPipeline' if in_channels == 9 else 'StableDiffusionXLPipeline'
    if in_channels == 9:
        return 'StableDiffusionInpaintPipeline'
    if in_channels == 5:
        return 'StableDiffusionDepth2ImgPipeline'
    return 'StableDiffusionPipeline'


def scan_model_folders(folders) -> list[ScannedModel]:
    results = []
    seen_paths = set()
    for root in folders:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]
            for name in filenames:
                if os.path.splitext(name)[1].lower() not in CHECKPOINT_EXTENSIONS:
                    continue
                path = os.path.join(dirpath, name)
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                model_type = classify_checkpoint(path)
                if model_type is None:
                    continue
                display = os.path.relpath(path, root).replace(os.sep, '/')
                results.append(ScannedModel(path, root, display, model_type))

    # disambiguate identical relative paths from different roots
    display_counts = {}
    for model in results:
        display_counts[model.display] = display_counts.get(model.display, 0) + 1
    for model in results:
        if display_counts[model.display] > 1:
            model.display = f"{os.path.basename(model.root)}/{model.display}"
    return results
