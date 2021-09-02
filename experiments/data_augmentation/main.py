import sys

sys.path.append('../../')
from experiment import RetinExp, DA, Dataset
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

    DA_test = [DA.NONE]

    if not isinstance(models, list):
        models = [models]

    for m in models:
        for d in DA_test:
            config['Manager']['run'] = '%s-DA: %s' % (m, d.name)
            config['Network']['architecture'] = m
            experiment = RetinExp(config, train_sets=Dataset.IDRID | Dataset.MESSIDOR,
                                  da_level=d,
                                  test_sets=Dataset.RETINAL_LESIONS | Dataset.IDRID | Dataset.DDR | Dataset.FGADR,
                                  cache=True)
            experiment.start()
