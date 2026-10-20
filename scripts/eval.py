"""Evaluate a SEED checkpoint or a Hugging Face model with lm-eval-harness.

Usage: see bash_scripts/evaluate_*.sh; configuration in configs/eval.yaml.
"""

import os

import accelerate
import hydra
import torch
from lm_eval.utils import make_table
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM

from src.checkpoint import load_model_from_run
from src.evaluation.lm_eval_adapter import LMEvalAdapter, build_stopping_criteria
from src.models.seed import SEED
from src.utils import count_parameters, format_number, register_resolvers, set_seed


def load_model(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    """`model_path` is a training run directory or a Hugging Face model id/path;
    `baseline` (eagle3 | dflash | dspark) adds that drafter to the Hugging Face
    model."""
    if cfg.get("baseline") is not None:
        return hydra.utils.instantiate(cfg.baseline, device=device)
    if os.path.exists(os.path.join(cfg.model_path, "config.yaml")):
        model = load_model_from_run(cfg.model_path, cfg.ckpt_file, device=device)
        if cfg.merge_lora and isinstance(model, SEED):
            model.drafter.merge_lora()
        return model
    kwargs = {} if cfg.torch_dtype is None else {"torch_dtype": cfg.torch_dtype}
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, trust_remote_code=True, **kwargs
    )
    return model.to(device)


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    accelerator = accelerate.Accelerator()
    if accelerator.num_processes == 1:
        accelerator = None
    set_seed(cfg.seed)
    device = (
        accelerator.device
        if accelerator is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    model = load_model(cfg, device)
    print(f"Num. params: {format_number(count_parameters(model, trainable=False))}")
    print(f"Num. trainable params: {format_number(count_parameters(model))}")
    print(f"Using tokenizer: {tokenizer.name_or_path}")

    adapter = LMEvalAdapter(
        model,
        tokenizer,
        generation_config=hydra.utils.instantiate(cfg.generation_config),
        max_new_tokens=cfg.max_new_tokens,
        stopping_criteria=build_stopping_criteria(
            list(cfg.stopping_criteria), tokenizer, cfg.max_new_tokens
        ),
        seed_decoding=OmegaConf.to_container(cfg.seed_decoding),
        chat_template_kwargs=OmegaConf.to_container(cfg.chat_template_kwargs),
        throughput_warmup=cfg.throughput_warmup,
        accelerator=accelerator,
    )
    results = hydra.utils.call(cfg.task, model=adapter)
    if results is None or (accelerator is not None and accelerator.local_process_index):
        return
    if isinstance(results, DictConfig):
        results = OmegaConf.to_container(results, resolve=True)
    print(make_table(results))
    if "groups" in results:
        print(make_table(results, "groups"))


if __name__ == "__main__":
    register_resolvers()
    main()
