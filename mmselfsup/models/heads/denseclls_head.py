# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.dist import get_rank
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS
from mmselfsup.utils import concat_all_gather


@MODELS.register_module()
class DenseCLHeadLs(BaseModule):
    """Head for MoCo v3 algorithms.

    This head builds a predictor, which can be any registered neck component.
    It also implements latent contrastive loss between two forward features.
    Part of the code is modified from:
    `<https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py>`_.

    Args:
        predictor (dict): Config dict for module of predictor.
        loss (dict): Config dict for module of loss functions.
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Defaults to 1.0.
    """

    def __init__(self,
                 predictor: dict,
                 loss: dict,
                 temperature: float = 1.0) -> None:
        super().__init__()
        self.predictor = MODELS.build(predictor)
        self.loss = MODELS.build(loss)
        self.temperature = temperature

    def forward(self, pos: torch.Tensor,
                neg: torch.Tensor) -> torch.Tensor:
        """Forward head.

        Args:
            base_out (torch.Tensor): NxC features from base_encoder.
            momentum_out (torch.Tensor): NxC features from momentum_encoder.

        Returns:
            torch.Tensor: The loss tensor.
        """
        # predictor computation
        pred = self.predictor([pos])[0]

        # normalize
        pos = nn.functional.normalize(pred, dim=1)
        neg = nn.functional.normalize(neg, dim=1)

        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).to(pos.device)

        loss = self.loss(logits, labels)
        return loss

