"""Train a SEED (or AR) model with Composer.

Usage: see bash_scripts/run_train_seed_tulu3_lora.sh; configuration in
configs/train.yaml.
"""

import logging

import hydra
import torch.distributed
from composer.utils import dist, reproducibility
from omegaconf import DictConfig, OmegaConf

from src.training import ComposerLM
from src.utils import (
    count_parameters,
    format_number,
    print_and_save_config,
    register_resolvers,
)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    print_and_save_config(cfg)
    reproducibility.seed_all(cfg.seed)

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    model = hydra.utils.instantiate(cfg.model, _convert_="all")
    print(model)
    model = ComposerLM(model)
    log.info(
        f"Num. parameters: {format_number(count_parameters(model, trainable=False))}"
    )
    log.info(f"Num. trainable parameters: {format_number(count_parameters(model))}")

    if not dist.is_initialized():
        dist.initialize_dist(timeout=600)
    collator = hydra.utils.instantiate(cfg.collator, tokenizer=tokenizer)

    def dataloader(dataset_cfg: DictConfig, loader_cfg: DictConfig, train: bool):
        dataset = hydra.utils.instantiate(dataset_cfg)
        return hydra.utils.instantiate(
            loader_cfg,
            _convert_="partial",
            dataset=dataset,
            collate_fn=collator,
            sampler=dist.get_sampler(dataset, shuffle=train, drop_last=train),
        )

    train_dataloader = dataloader(cfg.train_dataset, cfg.train_dataloader, train=True)
    eval_dataloader = dataloader(cfg.eval_dataset, cfg.eval_dataloader, train=False)

    composer_cfg = cfg.composer
    loggers = None
    if composer_cfg.loggers is not None:
        loggers = hydra.utils.instantiate(
            composer_cfg.loggers,
            _recursive_=False,
            _convert_="all",  # keeps the W&B config a plain dict
            init_kwargs={"config": OmegaConf.to_container(cfg, resolve=True)},
        )
    trainer = hydra.utils.instantiate(
        composer_cfg.trainer,
        _convert_="partial",
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizers=hydra.utils.instantiate(
            composer_cfg.optimizer, _convert_="all", params=model.parameters()
        ),
        schedulers=hydra.utils.instantiate(composer_cfg.lr_scheduler),
        algorithms=list(hydra.utils.instantiate(composer_cfg.algorithms).values()),
        loggers=loggers,
        callbacks=list(hydra.utils.instantiate(composer_cfg.callbacks).values()),
    )
    trainer.fit()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    register_resolvers()
    main()
