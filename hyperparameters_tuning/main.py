import sys

sys.path.append('../')
from experiment import RetinExp, DA, Dataset
from nntools.utils import Config
import argparse
import torch.nn as nn
from nntools.nnet import register_loss
import torch
import random


class MultiLabelSoftBinaryCrossEntropy(nn.Module):
    def __init__(self, smooth_factor=None):
        super(MultiLabelSoftBinaryCrossEntropy, self).__init__()
        self.smooth_factor = smooth_factor
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.smooth_factor is not None:
            smooth = random.uniform(0, self.smooth_factor)
            y_true = y_true.float()
            soft_targets = (1 - y_true) * smooth + y_true * (1 - smooth)
        else:
            soft_targets = y_true.float()
        B, C, H, W = y_pred.shape
        return sum([self.criterion(y_pred[:, i], soft_targets[:, i]) for i in range(C)])


register_loss('MultiLabelSoftBinaryCrossEntropy', MultiLabelSoftBinaryCrossEntropy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the configuration file")
    parser.add_argument("--models", help="List of models to train", nargs='+')
    parser.add_argument("--run_name", help="Name of the run")

    args = parser.parse_args()
    config_path = args.config
    models = args.models
    run_name = args.run_name
    config = Config(config_path)

    loss = ['Dice']
    if not isinstance(models, list):
        models = [models]

    for m in models:
        for l in loss:
            config['Manager']['run'] = '%s-%s' % (run_name, m)
            config['Network']['architecture'] = m
            config['Loss']['type'] = l
            experiment = RetinExp(config,
                                  DA_level=DA.COLOR, test_sets=Dataset.DDR | Dataset.IDRID | Dataset.RETINAL_LESIONS)
            experiment.eval()
