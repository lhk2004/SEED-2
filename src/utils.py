import os
import random

import hydra
import numpy as np
import rich.syntax
import rich.tree
import torch
from composer.utils import dist
from omegaconf import DictConfig, OmegaConf


def _world_size() -> int:
    if not dist.is_initialized():
        print("Initializing dist")
        dist.initialize_dist(timeout=600)
    return dist.get_world_size()


def _tokenizer_eos_token_id(tokenizer_cfg: DictConfig) -> int:
    return hydra.utils.instantiate(tokenizer_cfg).eos_token_id


def register_resolvers() -> None:
    """OmegaConf resolvers used by the configs."""
    OmegaConf.register_new_resolver("cwd", os.getcwd)
    OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)
    OmegaConf.register_new_resolver(
        "device", lambda: "gpu" if torch.cuda.is_available() else "cpu"
    )
    OmegaConf.register_new_resolver("world_size", _world_size)
    OmegaConf.register_new_resolver("tokenizer_eos_token_id", _tokenizer_eos_token_id)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module, trainable: bool = True) -> int:
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable
    )


def format_number(num: int) -> str:
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(num) >= threshold:
            return f"{num / threshold:.1f}{suffix}"
    return str(num)


def print_and_save_config(cfg: DictConfig, save_dir: str | None = None) -> None:
    """Print the resolved config as a tree; save it to `config.yaml` (which the
    evaluation script reads) and `config_tree.txt`."""
    tree = rich.tree.Tree("CONFIG", style="dim", guide_style="dim")
    for field in cfg.keys():
        branch = tree.add(field, style="dim", guide_style="dim")
        section = cfg.get(field)
        content = (
            OmegaConf.to_yaml(section, resolve=True)
            if isinstance(section, DictConfig)
            else str(section)
        )
        branch.add(rich.syntax.Syntax(content, "yaml"))
    rich.print(tree)
    save_dir = save_dir or os.getcwd()
    with open(os.path.join(save_dir, "config_tree.txt"), "w") as f:
        rich.print(tree, file=f)
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f, resolve=True)
