import torch.nn as nn
from nntools.nnet import AbstractNet
import segmentation_models_pytorch as smp
import torch
from nntools.nnet import Conv2d


class ContrastiveUnet(AbstractNet):
    def __init__(self, classes=4, encoder_weights='imagenet', contrastive_features=128):
        super(ContrastiveUnet, self).__init__()
        self.model = smp.Unet('resnet101', classes=classes, encoder_weights=encoder_weights)
        encoder_out_channel = self.model.encoder.out_channels[-1]

        self.contrastive_head = nn.Sequential(Conv2d(encoder_out_channel, 256, bias=True), Conv2d(256, 256, bias=True),
                                              Conv2d(256, contrastive_features, bias=True, activation=nn.Identity()))
        self.segmentation_mode = False

    def activate_segmentation_mode(self):
        self.segmentation_mode = True

    def activate_contrastive_mode(self):
        self.segmentation_mode = False

    def forward(self, x):
        if self.segmentation_mode:
            return self.model(x)
        else:
            features = self.encoder(x)
            contrastive_projection = self.contrastive_head(features[-1])
            return contrastive_projection / torch.norm(contrastive_projection, 0, keepdim=True)




