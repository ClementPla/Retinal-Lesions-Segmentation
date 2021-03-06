import sys

sys.path.append('../')
from experiment import RetinExp
from nntools.utils import Config
import argparse
from scripts.utils import DA, Dataset

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
    cache = False
    for m in models:
        config['Network']['architecture'] = m
        train_set = Dataset.IDRID | Dataset.MESSIDOR
        config['Manager']['run'] = '%s-%s' % (train_set.str_name, m)
        experiment = RetinExp(config, da_level=DA.COLOR,
                              train_sets=train_set,
                              test_sets=Dataset.RETINAL_LESIONS | Dataset.IDRID | Dataset.DDR,
                              cache=cache)
        experiment.register_params = False
        experiment.start()
