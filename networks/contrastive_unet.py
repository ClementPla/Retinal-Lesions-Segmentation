import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from nntools.nnet import AbstractNet
from nntools.nnet import Conv2d
from copy import deepcopy
from icecream import ic


class ContrastiveUnet(AbstractNet):
    def __init__(self, classes=4, encoder_weights='imagenet', contrastive_features=128):
        super(ContrastiveUnet, self).__init__()
        self.model = smp.Unet('resnet101', classes=classes, encoder_weights=encoder_weights)
        self.regular_segmentation_head = self.model.segmentation_head

        self.contrastive_head = nn.Sequential(Conv2d(16, 256, bias=True), Conv2d(256, 256, bias=True),
                                              Conv2d(256, contrastive_features, bias=True, activation=nn.Identity()))

        self.model.segmentation_head = self.contrastive_head
        self.segmentation_mode = False

    def activate_segmentation_mode(self):
        self.segmentation_mode = True
        self.model.segmentation_head = self.regular_segmentation_head

    def activate_contrastive_mode(self):
        self.segmentation_mode = False
        self.model.segmentation_head = self.contrastive_head

    def forward(self, x):
        if self.segmentation_mode:
            return self.model(x)
        else:
            contrastive_projection = self.model(x)
            return contrastive_projection / torch.norm(contrastive_projection, dim=1, keepdim=True)
