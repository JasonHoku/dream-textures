import os

from .huggingface_hub import DownloadStatus
from ..future import Future
from ..models import ModelConfig


def convert_original_stable_diffusion_to_diffusers(
    self,
    checkpoint_path: str,
    model_config: ModelConfig,
    half_precision: bool,
) -> str:
    import torch
    from huggingface_hub.constants import HF_HUB_CACHE

    future = Future()
    yield future
    DownloadStatus.hook_download_tqdm(future)

    future.add_response(DownloadStatus(f"Reading {checkpoint_path}", 0, 1))
    index = 0
    def hook_save_pretrained(model, dirs_count, total):
        old_save_pretrained = model.save_pretrained
        def save_pretrained(self, save_directory, *args, **kwargs):
            nonlocal index
            dirs = []
            directory = save_directory
            for _ in range(dirs_count):
                dirs.append(os.path.basename(directory))
                directory = os.path.dirname(directory)
            dirs.reverse()
            future.add_response(DownloadStatus(f"Saving {os.path.join(*dirs)}", index, total))
            index += 1
            return old_save_pretrained(save_directory, *args, **kwargs)
        model.save_pretrained = save_pretrained.__get__(model)

    original_config_kwargs = {}
    if model_config.original_config is not None:
        original_config_kwargs["original_config"] = model_config.original_config

    if model_config in [ModelConfig.CONTROL_NET_1_5, ModelConfig.CONTROL_NET_2_1]:
        from diffusers import ControlNetModel
        pipe = ControlNetModel.from_single_file(
            checkpoint_path,
            **original_config_kwargs,
        )
        if half_precision:
            pipe.to(dtype=torch.float16)
        index = 1
        hook_save_pretrained(pipe, 1, 2)
    else:
        pipe = model_config.pipeline.from_single_file(
            checkpoint_path,
            **original_config_kwargs,
        )
        if half_precision:
            pipe.to(torch_dtype=torch.float16)
        models = []
        for name in pipe._get_signature_keys(pipe)[0]:
            model = getattr(pipe, name, None)
            if model is not None and hasattr(model, "save_pretrained"):
                models.append(model)
        for i, model in enumerate(models):
            hook_save_pretrained(model, 2, len(models))
    dump_path = os.path.join(HF_HUB_CACHE, os.path.splitext(os.path.basename(checkpoint_path))[0])
    pipe.save_pretrained(dump_path, variant="fp16" if half_precision else None)
    future.set_done()
