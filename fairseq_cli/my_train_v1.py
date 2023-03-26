"""Implementation of fairseq_cli/train.py"""
import argparse
import logging
import math
import os
import sys
from typing import Tuple, List, Optional

from fairseq.dataclass import FairseqDataclass

from fairseq.criterions import FairseqCriterion
from fairseq.models import FairseqModel
from fairseq.tasks import FairseqTask
from fairseq_cli.train import get_training_stats, should_stop_early, get_valid_stats

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from fairseq import checkpoint_utils, options, quantization_utils, tasks, utils
from fairseq.data import data_utils, iterators, EpochBatchIterator, CountingIterator
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import add_defaults
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap
from fairseq.distributed import utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer


def main(cfg: FairseqConfig):
    """main function of train"""
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)
    """1. reset_metrics"""
    metrics.reset()
    """2. set random seed"""
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)
    """3. master check ckp dir"""
    if distributed_utils.is_master(cfg=cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)
    """4. setup_task"""
    task: FairseqTask = tasks.setup_task(cfg.task)
    """5. build_model and setup criterion"""
    model: FairseqModel = task.build_model(cfg.model)
    criterion: FairseqCriterion = task.build_criterion(cfg=cfg.criterion)
    """6. load valid_dataset first"""
    if cfg.dataset.combine_valid_subsets:
        task.load_dataset(split="valid", combine=True, task_cfg=cfg.task, epoch=1)
    else:
        for subset in cfg.dataset.valid_subset.split(","):
            task.load_dataset(split=subset, combine=False, task_cfg=cfg.task, epoch=1)
    """7. build trainer"""
    trainer = Trainer(cfg=cfg, task=task, model=model, criterion=criterion)
    """8. load ckp"""
    extra_stat, epoch_itr = checkpoint_utils.load_checkpoint(cfg=cfg.checkpoint, trainer=trainer)
    """9. load max_epoch & set lr"""
    max_epoch: int = cfg.optimization.max_epoch or math.inf
    lr: float = trainer.get_lr()
    """10. start training"""
    while epoch_itr.next_epoch_idx <= max_epoch:
        """11. check lr"""
        if lr <= cfg.optimization.stop_min_lr:
            break
        """12. train one epoch"""
        valid_losses, should_stop = train(cfg=cfg, trainer=trainer, task=task, epoch_itr=epoch_itr)
        """13. update lr"""
        if len(valid_losses) > 0:
            trainer.lr_step(epoch=epoch_itr.epoch, val_loss=valid_losses[0])
        """14. get new batch itr"""
        epoch_itr = trainer.get_train_iterator(epoch=epoch_itr.next_epoch_idx, combine=True,
                                               load_dataset=task.has_sharded_data("train"))


def train(cfg: DictConfig, trainer: Trainer, task: FairseqTask, epoch_itr: EpochBatchIterator) \
        -> Tuple[List[Optional[float]], bool]:
    """train one epoch"""
    """1. init_epoch_itr"""
    itr: CountingIterator = epoch_itr.next_epoch_itr(shuffle=(cfg.dataset.curriculum < epoch_itr.next_epoch_idx),
                                                     fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus)
    """2. get update_freq"""
    update_freq: int = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1] if epoch_itr.next_epoch_idx <= len(
            cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )

    """3. get grouped itr"""
    batches_itr = iterators.GroupedIterator(iterable=itr, chunk_size=update_freq,
                                            skip_remainder_batch=cfg.optimization.skip_remainder_batch)
    """4. get progress bar"""
    progress = progress_bar.progress_bar(iterator=batches_itr, log_format=cfg.common.log_format,
                                         log_interval=cfg.common.log_interval, log_file=cfg.common.log_file,
                                         epoch=epoch_itr.epoch, aim_param_checkpoint_dir=cfg.checkpoint.save_dir)
    """5. update config"""
    progress.update_config(config=cfg)
    """6. trainer begin epoch"""
    trainer.begin_epoch(epoch=epoch_itr.epoch)
    """7. get valid subsets, num_update"""
    valid_subsets = cfg.dataset.valid_subset.split(",")
    num_updates = trainer.get_num_updates()
    should_stop = False
    valid_losses = []
    """8. train one epoch"""
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function("train_step_%d" % i):
            logging_output = trainer.train_step(samples=samples)
        """9. log mid epoch stats"""
        if logging_output is not None:
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats=stats, tag="train_inner", step=num_updates)
        """10. reset meters"""
        metrics.reset_meters(name="train_inner")
        """11. valid and save"""
        valid_losses, should_stop = validate_and_save(cfg=cfg, trainer=trainer, task=task, epoch_itr=epoch_itr,
                                                      valid_subsets=valid_subsets,
                                                      end_of_epoch=epoch_itr.end_of_epoch())
        if should_stop:
            break
    """12. log end epoch stats"""
    progress.log(stats=get_training_stats(metrics.get_smoothed_values("train")), tag="train", step=num_updates)
    """13. reset meters"""
    metrics.reset_meters(name="train")
    return valid_losses, should_stop


def validate_and_save(cfg: DictConfig, trainer: Trainer, task: FairseqTask, epoch_itr: EpochBatchIterator,
                      valid_subsets: list, end_of_epoch: bool) -> Tuple[Optional[List], bool]:
    should_stop = False
    num_update = trainer.get_num_updates()
    """1. check stop: num_updates > max_updates"""
    if trainer.get_num_updates() > cfg.optimization.max_update:
        logger.info(msg=f"stop due to num_updates={trainer.get_num_updates()} "
                        f"> max_updates{cfg.optimization.max_update}")
        should_stop = True
    """2. check stop: training_hour > stop_hour"""
    if trainer.cumulative_training_time() / (60 * 60) > cfg.optimization.stop_time_hours:
        logger.info(msg=f"stop due to cumulative_training_time={trainer.cumulative_training_time() / (60 * 60)} "
                        f"> stop_time_hours{cfg.optimization.stop_time_hours}")
        should_stop = True
    """3. check do save"""
    do_save = (
            (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)  # save every interval epoch
            or should_stop  # 停止训练
            or (
                    cfg.checkpoint.save_interval_updates > 0
                    and num_update > 0
                    and num_update % cfg.checkpoint.save_interval_updates == 0
                    and num_update >= cfg.dataset.validate_after_updates
            )  # 间隔update保存
    )
    """4. check do validate"""
    do_validate = (
            ((not end_of_epoch and do_save)
             or (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
             or should_stop
             or (
                     cfg.dataset.validate_interval_updates > 0
                     and num_update > 0
                     and num_update % cfg.dataset.validate_interval_updates == 0
             ))
            and not cfg.dataset.disable_validation
            and num_update >= cfg.dataset.validate_after_updates
    )
    """5. do validate"""
    valid_losses = []
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)
    """6. do save"""
    should_stop |= should_stop_early(cfg, valid_losses[0])
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(cfg=cfg.checkpoint, trainer=trainer, epoch_itr=epoch_itr,
                                         val_loss=valid_losses[0])
        return valid_losses, should_stop

def validate(cfg: DictConfig, trainer: Trainer, task: FairseqTask,
             epoch_itr: EpochBatchIterator, valid_subsets: List) -> Optional[List]:
    """1. trainer begin epoch"""
    trainer.begin_epoch(epoch=epoch_itr.epoch)
    valid_losses = []
    """2. valid on each subset"""
    for subset_idx, subset in enumerate(valid_subsets):
        itr: EpochBatchIterator = trainer.get_valid_iterator(subset=subset)
        itr = itr.next_epoch_itr(shuffle=False, set_dataset_epoch=False)
        """3. get progress bar"""
        progress = progress_bar.progress_bar(iterator=itr, log_format=cfg.common.log_format,
                                             epoch=epoch_itr.epoch,
                                             prefix=f"valid on '{subset}' subset")
        with metrics.aggregate(new_root=True) as agg:
            """4. valid on one subset"""
            for i, sample in enumerate(progress):
                if cfg.dataset.max_valid_steps is not None and i > cfg.dataset.max_valid_steps:
                    break
                trainer.valid_step(sample=subset,)
        """5. log valid stats"""
        tracking_best = subset_idx == 0
        stats = get_valid_stats(cfg=cfg, trainer=trainer, stats=agg.get_smoothed_values(),
                                tracking_best=tracking_best)

        if hasattr(task, "post_validate"):
            task.post_validate(trainer.get_model(), stats, agg)

        progress.print(stats=stats, tag=subset, step=trainer.get_num_updates())
        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses

