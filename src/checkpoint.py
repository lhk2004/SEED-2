"""Load trained models from training run directories.

A run directory holds the training `config.yaml` and a `checkpoints/` folder with
Composer checkpoints. The model weights extracted from a checkpoint are cached
next to it as `<name>_weights_only.pt`.

Runs saved before the codebase was restructured (`src.denoiser.diffusion.SEED`
with a `LLMasEncoderDecoderFrozenTargetLoRA` backbone) are upgraded on load.
"""

import os
from typing import Any

import hydra
import torch
import yaml
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

_LEGACY_SEED = "src.denoiser.diffusion.SEED"
_LEGACY_LORA_BACKBONE = "src.backbone.seed_lora.LLMasEncoderDecoderFrozenTargetLoRA"
_LEGACY_KEY_PREFIXES = (
    ("backbone.encoder.", "target."),
    ("backbone.decoder.model.", "drafter."),
    ("backbone.decoder.lm_head.", "drafter.lm_head."),
)
# Attention masks that used to be saved as buffers.
_LEGACY_BUFFERS = ("static_attention_mask", "encoder_static_attention_mask")


def _upgrade_legacy_model_config(model_cfg: dict[str, Any]) -> dict[str, Any]:
    target = model_cfg.get("_target_", "")
    if not target.startswith("src.denoiser."):
        return model_cfg
    config = model_cfg["config"]
    backbone = config["backbone_config"]
    if target != _LEGACY_SEED or backbone["_target_"] != _LEGACY_LORA_BACKBONE:
        raise ValueError(
            f"Unsupported legacy checkpoint ({target}); only SEED-LoRA runs load."
        )
    return {
        "_target_": "src.models.seed.SEED",
        "pretrained_model_name_or_path": backbone["pretrained_model_name_or_path"],
        "max_seq_length": config["length"],
        "block_size": config["block_size"],
        "num_draft_layers": backbone.get("num_decoder_layers", 2),
        "lora_rank": backbone.get("lora_rank", 32),
        "lora_alpha": backbone.get("lora_alpha", 64.0),
        "lora_dropout": backbone.get("lora_dropout", 0.0),
        "torch_dtype": backbone.get("torch_dtype"),
        "train_on_context": config.get("train_on_context", False),
    }


def _upgrade_legacy_state_dict(state_dict: dict[str, torch.Tensor]) -> None:
    for name in _LEGACY_BUFFERS:
        state_dict.pop(name, None)
    for key in list(state_dict):
        for old, new in _LEGACY_KEY_PREFIXES:
            if key.startswith(old):
                state_dict[new + key[len(old) :]] = state_dict.pop(key)
                break


def _load_weights(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    _upgrade_legacy_state_dict(state_dict)
    # Frozen weights must equal the freshly loaded pretrained ones; otherwise the
    # checkpoint was trained against a different base model.
    for name, param in model.named_parameters(remove_duplicate=False):
        if (
            not param.requires_grad
            and name in state_dict
            and not torch.equal(param.detach(), state_dict[name].to(param))
        ):
            raise ValueError(
                f"Checkpoint changed frozen weight {name}. Use the original base "
                "model revision and an unmodified checkpoint."
            )
    # Missing adapters must never silently evaluate an untrained drafter.
    model.load_state_dict(state_dict, strict=True)


def load_model_from_run(
    run_dir: str,
    ckpt_file: str = "best-rank0.pt",
    device: torch.device | str = "cpu",
) -> torch.nn.Module:
    """Build the model described by `run_dir/config.yaml` and load its weights."""
    with open(os.path.join(run_dir, "config.yaml")) as f:
        config = yaml.safe_load(f)
    model = hydra.utils.instantiate(
        _upgrade_legacy_model_config(config["model"]), _convert_="all"
    )

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    weights_path = os.path.join(
        checkpoint_dir, f"{ckpt_file.removesuffix('.pt')}_weights_only.pt"
    )
    if os.path.exists(weights_path):
        print(f"Loading cached weights-only checkpoint from: {weights_path}")
        _load_weights(model, torch.load(weights_path, map_location="cpu"))
    else:
        checkpoint = torch.load(
            os.path.join(checkpoint_dir, ckpt_file),
            weights_only=False,
            map_location="cpu",
            mmap=True,
        )
        state_dict = checkpoint["state"]["model"]
        for prefix in ("module.", "model."):  # DDP / ComposerLM wrappers
            consume_prefix_in_state_dict_if_present(state_dict, prefix)
        _load_weights(model, state_dict)
        print(f"Saving extracted weights to {weights_path}")
        torch.save(state_dict, weights_path)
        del checkpoint, state_dict
    return model.to(device)
