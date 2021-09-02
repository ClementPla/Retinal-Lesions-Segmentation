import sys
sys.path.append('../../')
from experiment import RetinExp
from nntools.utils import Config
import argparse
from tools.utils import DA, Dataset
import torch
torch.backends.cudnn.benchmark = True


def get_exp(config, train_set, cache, run_id=None):
    experiment = RetinExp(config, da_level=DA.COLOR | DA.GEOMETRIC_SIMPLE,
                          run_id=run_id,
                          train_sets=train_set,
                          test_sets=Dataset.IDRID | Dataset.DDR | Dataset.RETINAL_LESIONS,
                          cache=cache)
    return experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the configuration file")

    args = parser.parse_args()
    config_path = args.config
    config = Config(config_path)

    cache = False
    train_set = Dataset.MESSIDOR | Dataset.IDRID | Dataset.FGADR
    config['Manager']['run'] = '%s' % train_set.str_name
    exp = get_exp(config, train_set, cache)
    model_path = config['Training']['MultiDataset']['initial_model_path']
    print(exp.model.load(model_path, filtername='best', load_most_recent=True, allow_size_mismatch=True))
    exp.start()

