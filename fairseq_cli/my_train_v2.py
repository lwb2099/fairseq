"""
v2 of my implementation
Date: 2023-3-24
"""
import argparse
import logging
import os
import sys
from typing import *
import numpy as np
import torch
import math

from fairseq.data import iterators

from fairseq.criterions import FairseqCriterion
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import *
from fairseq.logging import metrics, progress_bar
from fairseq.models import build_model, FairseqModel
from fairseq.tasks import setup_task, FairseqTask
from fairseq.trainer import Trainer
from fairseq.utils import *
from fairseq.distributed.utils import *
from fairseq.checkpoint_utils import *
from fairseq_cli.train import get_training_stats, should_stop_early, get_valid_stats

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


def main(cfg: FairseqConfig):
    """main func of fairseq_cli / train.py"""
    if isinstance(cfg, argparse.Namespace):
        cfg: DictConfig = convert_namespace_to_omegaconf(cfg)
    """1. reset metrics"""
    metrics.reset()
    """2. set random seed"""
    np.random.seed(cfg.common.seed)
    set_torch_seed(seed=cfg.common.seed)
    """3. check ckp dir"""
    if is_master(cfg=cfg.distributed_training):
        verify_checkpoint_directory(cfg.checkpoint.save_dir)
    """4. setup task"""
    task: FairseqTask = setup_task(cfg=cfg.task)
    """5. build model & cri"""
    model: FairseqModel = build_model(cfg=cfg.model, task=task)
    criterion: FairseqCriterion = task.build_criterion(cfg=cfg.criterion)
    """6. load valid dataset"""
    if cfg.dataset.combine_valid_subsets:
        task.load_dataset(split="valid", combine=True, task_cfg=cfg.task, epoch=1)
    else:
        for split in cfg.dataset.valid_subset.split(","):
            task.load_dataset(split=split, combine=False, task_cfg=cfg.task, epoch=1)
    """7. build trainer"""
    trainer: Trainer = Trainer(cfg=cfg, task=task, model=model, criterion=criterion)
    """8. load ckp"""
    extra_stat, epoch_itr = load_checkpoint(cfg=cfg.checkpoint, trainer=trainer)
    """9. setup lr & load max_epoch"""
    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()
    """10. begin training"""
    while epoch_itr.next_epoch_idx <= max_epoch:
        """11. check lr"""
        if lr < cfg.optimization.stop_min_lr:
            break
        """12. train for one epoch"""
        validate_losses, should_stop = train(cfg=cfg, trainer=trainer, task=task, epoch_itr=epoch_itr)
        """13. update lr"""
        lr = trainer.lr_step(epoch=epoch_itr.epoch, val_loss=validate_losses[0])
        """14. get new itr"""
        epoch_itr = trainer.get_train_iterator(epoch=epoch_itr.next_epoch_idx, combine=True,
                                               load_dataset=task.has_sharded_data(split="train"))


def train(cfg: DictConfig, trainer: Trainer, task: FairseqTask, epoch_itr: EpochBatchIterator) -> Tuple[
    Optional[list], bool]:
    """train for one epoch"""
    """1. get new itr"""
    itr = epoch_itr.next_epoch_itr(shuffle=(cfg.dataset.curriculum < epoch_itr.next_epoch_idx),
                                   fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus)
    """2. get update_freq"""
    update_freq = cfg.optimization.update_freq[epoch_itr.next_epoch_idx - 1] \
        if epoch_itr.next_epoch_idx <= len(cfg.optimization.update_freq) \
        else cfg.optimization.update_freq[-1]
    """3. get batch itr"""
    batch_itr = iterators.GroupedIterator(iterable=itr, chunk_size=update_freq,
                                          skip_remainder_batch=cfg.optimization.skip_remainder_batch)
    """4. get progress_bar"""
    progress = progress_bar.progress_bar(iterator=batch_itr, )
    """5. update config"""
    progress.update_config()
    """6. trainer.begin_epoch"""
    trainer.begin_epoch(epoch=epoch_itr.epoch)
    """7. get num_update & valid_subsets"""
    num_update = trainer.get_num_updates()
    valid_subsets = cfg.dataset.valid_subset.split(",")
    valid_losses, should_stop = [], False
    """8. train one epoch"""
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function("train_step_%d" % i):
            logging_outputs = trainer.train_step(samples=samples)
        """9. log mid epoch stats"""
        if logging_outputs is not None:
            num_update = trainer.get_num_updates()
            if num_update % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats=stats, tag="train_inner", step=num_update)
        """10. reset meters"""
        metrics.reset_meters("train_inner")
        """11. valid and save"""
        valid_losses, should_stop = validate_and_save(cfg=cfg, trainer=trainer, task=task, epoch_itr=epoch_itr,
                                                      valid_subsets=valid_subsets,
                                                      end_of_epoch=epoch_itr.end_of_epoch())
    """12. log end epoch stats"""
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.log(stats=stats, tag="train", step=num_update)
    """13. reset meters"""
    metrics.reset_meters("train")
    return valid_losses, should_stop


def validate_and_save(cfg: DictConfig, trainer: Trainer, task: FairseqTask, epoch_itr: EpochBatchIterator,
                      valid_subsets: List, end_of_epoch: bool) -> Tuple[Optional[List], bool]:
    """1. check stop: num_update > max_update"""
    num_update = trainer.get_num_updates()
    should_stop = False
    if num_update > cfg.optimization.max_update:
        logger.info(msg=f"stop training due to num_update={num_update} > max_update={cfg.optimization.max_update}")
        should_stop = True
    """2. check stop: train_hour > stop_train_hour"""
    if trainer.cumulative_training_time() > cfg.optimization.stop_time_hours:
        logger.info(msg=f"stop training due to {trainer.cumulative_training_time()} > "
                        f"{cfg.optimization.stop_time_hours}")
        should_stop = True
    """3. check do save"""
    do_save = (
            # each N epoch
            (not end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
            or should_stop
            or (
                    num_update > 0 and num_update % cfg.checkpoint.save_interval_updates == 0 and
                    num_update >= cfg.dataset.validate_after_updates
            )  # each N update
    )
    """4. check do validate"""
    do_validate = (
        # mid_epoch & do_save
        (not end_of_epoch and do_save)
        or should_stop
        or (
            cfg.dataset.validate_interval_updates > 0 and num_update > 0 and
            num_update % cfg.dataset.validate_interval_updates == 0
        )
        and not cfg.dataset.disable_validation
        and (num_update >= cfg.dataset.validate_after_updates)
    )
    """5. do validate"""
    valid_losses = []
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)
    """6. do save"""
    should_stop |= should_stop_early(cfg=cfg, valid_loss=valid_losses[0])
    if do_save or should_stop:
        save_checkpoint(cfg=cfg.checkpoint, trainer=trainer, epoch_itr=epoch_itr, val_loss=valid_losses[0])
    return valid_losses, should_stop

def validate(cfg: DictConfig, trainer: Trainer, task: FairseqTask, epoch_itr: EpochBatchIterator,
             valid_subsets: List) -> Optional[List]:
    """1. begin valid epoch"""
    trainer.begin_valid_epoch(epoch=epoch_itr.epoch)
    valid_losses = []
    for subset_idx, subset in enumerate(valid_subsets):
        """2. get itr"""
        itr = trainer.get_valid_iterator(subset=subset)
        """3. get progress bar"""
        progress = progress_bar.progress_bar(
            iterator=itr, log_format=cfg.common.log_format, log_interval=cfg.common.log_interval,
            log_file=cfg.common.log_file, epoch=epoch_itr.epoch, prefix=f"valid on '{subset}' subset",
        )
        """4. valid one subset"""
        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                if cfg.dataset.max_valid_steps is not None and i > cfg.dataset.max_valid_steps:
                    trainer.valid_step(sample=sample)
        """5. log valid stats"""
        tracking_best = subset_idx == 0
        stats = get_valid_stats(cfg=cfg, trainer=trainer, stats=agg.get_smoothed_values(), tracking_best=tracking_best)
        if hasattr(task, "post_valid"):
            task.post_valid(trainer.get_model(), stats, agg)
        progress.print(stats=stats, tag=subset, step=trainer.get_num_updates())
        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses

