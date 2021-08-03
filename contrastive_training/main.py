import sys

sys.path.append('../')
from experiment import RetinExp, DA, Dataset, ContrastiveLoss
from nntools.utils import Config
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the configuration file")
    parser.add_argument("--models", help="List of models to train", nargs='+')

    args = parser.parse_args()
    config_path = args.config
    models = args.models
    config = Config(config_path)

    if not isinstance(models, list):
        models = [models]

    for m in models:
        config['Manager']['run'] = '%s' % m
        config['Network']['architecture'] = m
        experiment = RetinExp(config, train_sets=Dataset.IDRID | Dataset.MESSIDOR | Dataset.FGADR,
                              contrastive_loss=ContrastiveLoss.SINGLE_IMAGE,
                              DA_level=DA.GEOMETRIC,
                              test_sets=Dataset.RETINAL_LESIONS | Dataset.IDRID | Dataset.DDR,
                              cache=True)
        experiment.start()

        experiment = RetinExp(config, train_sets=Dataset.IDRID | Dataset.MESSIDOR | Dataset.FGADR,
                              DA_level=DA.GEOMETRIC,
                              contrastive_loss=ContrastiveLoss.CROSS_IMAGES,
                              test_sets=Dataset.RETINAL_LESIONS | Dataset.IDRID | Dataset.DDR,
                              cache=True)
        experiment.start()

