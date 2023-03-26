"""Date: 2023-3-24"""
import torch

from fairseq.criterions import FairseqCriterion
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel
from fairseq.optim import FairseqOptimizer

from fairseq.tasks.fairseq_task import StatefulContainer


class MyFairseqTask(object):
    def __init__(self, cfg: FairseqDataclass, **kwargs):
        self.cfg = cfg
        self.datasets = dict()
        self.dataset_to_epoch_iter = dict()
        self.state = StatefulContainer()

    def train_step(self, sample: dict, model: BaseFairseqModel, criterion: FairseqCriterion,
                   optimizer: FairseqOptimizer, update_num: int, ignore_grad: bool = False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True
        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        """1. train&set_num_update"""
        model.train()
        model.set_num_updates(num_updates=update_num)
        with torch.autograd.profiler.record_function("forward"):
            """2. forward"""
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        """3. backward"""
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output


    def valid_step(self, sample: dict, model: BaseFairseqModel, criterion: FairseqCriterion):
        model.eval()
        with torch.no_grad():
            valid_loss, sample_size, logging_output = criterion(model, sample)
        return valid_loss, sample_size, logging_output