import sys

sys.path.append('../../')
from experiment import RetinExp
from nntools.utils import Config
import argparse
from tools.utils import DA, Dataset

def get_exp(config, train_set, cache, run_id=None):
    experiment = RetinExp(config, da_level=DA.COLOR | DA.GEOMETRIC_SIMPLE,
                          run_id=run_id,
                          train_sets=train_set,
                          test_sets=Dataset.RETINAL_LESIONS | Dataset.IDRID | Dataset.DDR,
                          cache=cache)
    return experiment


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
        #
        # train_set = Dataset.IDRID
        # config['Manager']['run'] = '%s-%s' % (train_set.str_name, m)
        # get_exp(config, train_set, cache, run_id='27107dec8eb049cab730a89c1c334ffb').eval()
        #
        # train_set = Dataset.IDRID | Dataset.MESSIDOR
        # config['Manager']['run'] = '%s-%s' % (train_set.str_name, m)
        # get_exp(config, train_set, cache).start()

        # train_set = Dataset.IDRID | Dataset.MESSIDOR | Dataset.FGADR
        # config['Manager']['run'] = '%s-%s' % (train_set.str_name, m)
        # get_exp(config, train_set, cache).start()
        #
        # train_set = Dataset.FGADR
        # config['Manager']['run'] = '%s-%s' % (train_set.str_name, m)
        # get_exp(config, train_set, cache).start()
        #
        # train_set = Dataset.MESSIDOR
        # config['Manager']['run'] = '%s-%s' % (train_set.str_name, m)
        # get_exp(config, train_set, cache).start()
        #
        # train_set = Dataset.FGADR | Dataset.IDRID
        # config['Manager']['run'] = '%s-%s' % (train_set.str_name, m)
        # get_exp(config, train_set, cache).start()

        # TODO : train_set = Dataset.FGADR | Dataset.MESSIDOR

        train_set = Dataset.FGADR | Dataset.MESSIDOR
        config['Manager']['run'] = '%s-%s' % (train_set.str_name, m)
        get_exp(config, train_set, cache).start()