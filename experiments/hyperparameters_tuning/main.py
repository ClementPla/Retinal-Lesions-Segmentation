import sys

sys.path.append('../../')
from experiment import RetinExp, DA, Dataset
from nntools.utils import Config
import argparse


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

    loss = ['NLL']
    lrs = [1e-4]
    wds = [1e-5]

    if not isinstance(models, list):
        models = [models]

    for m in models:
        config['Manager']['run'] = '%s-%s' % (run_name, m)
        config['Network']['architecture'] = m
        for l in loss:
            for lr in lrs:
                for wd in wds:
                    config['Network']['architecture'] = m
                    config['Optimizer']['params_solver']['lr'] = lr
                    config['Optimizer']['params_solver']['weight_decay'] = wd
                    config['Loss']['type'] = l
                    experiment = RetinExp(config,
                                          train_sets=Dataset.IDRID,
                                          da_level=DA.COLOR | DA.GEOMETRIC_SIMPLE, test_sets=Dataset.IDRID)
                    experiment.start()
