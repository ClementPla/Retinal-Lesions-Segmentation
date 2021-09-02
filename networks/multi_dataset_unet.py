import segmentation_models_pytorch as smp
import torch.nn as nn
from nntools.nnet import AbstractNet
from nntools.nnet import Conv2d

from tools.utils import Dataset


class MultiDatasetUnet(AbstractNet):
    def __init__(self, datasets, classes=4, encoder_name='resnet34', encoder_weights='imagenet'):
        super(MultiDatasetUnet, self).__init__()
        self.network = smp.Unet(encoder_name, classes=classes, encoder_weights=encoder_weights)
        self.regular_segmentation_head = self.network.segmentation_head

        self.c = classes
        self.datasets = []
        self.cm_predictors = nn.ModuleDict()

        def fill_cm_predictors(d):
            if d in datasets:
                self.datasets.append(d)
                self.cm_predictors[d.name] = nn.Sequential(Conv2d(16, 256, bias=True),
                                                           Conv2d(256, 256, bias=True),
                                                           Conv2d(256, (classes+1)*(classes+1),
                                                                  activation=nn.ReLU()))

        fill_cm_predictors(Dataset.IDRID)
        fill_cm_predictors(Dataset.MESSIDOR)
        fill_cm_predictors(Dataset.RETINAL_LESIONS)
        fill_cm_predictors(Dataset.DDR)
        fill_cm_predictors(Dataset.FGADR)

    def forward(self, x, tag=None):
        """

        :param x: Tensor of size BxCxHxW
        :param tag: Tensor of size B
        :return:
        """
        x = self.network.encoder(x)
        x = self.network.decoder(*x)
        y = self.network.segmentation_head(x)

        if tag is not None:
            cm_predictions = []
            for d in self.datasets:
                in_tensor = x[tag == d.value]  # Get the batch corresponding to dataset i
                if in_tensor.shape[0]:

                    cm = self.cm_predictors[d.name](in_tensor)

                    cm_predictions.append((d.value, cm))
            return y, cm_predictions, tag
        else:
            return y

    def grads_control(self, encoder=True, decoder=True, segmentation_head=True, cm_predictors=True):
        for p in self.network.encoder.parameters(): p.requires_grad = encoder
        for p in self.network.decoder.parameters(): p.requires_grad = decoder
        for p in self.network.segmentation_head.parameters(): p.requires_grad = segmentation_head
        for p in self.cm_predictors.parameters(): p.requires_grad = cm_predictors
