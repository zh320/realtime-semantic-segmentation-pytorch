import argparse
import importlib
import warnings

from core import SegTrainer
from configs import MyConfig, load_parser, load_config_json

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--config-json", type=str, default=None)
    args, remaining = parser.parse_known_args()

    if args.config is not None:
        module_name = args.config.replace("/", ".").removesuffix(".py")
        module = importlib.import_module(module_name)
        config = module.MyConfig()
    else:
        config = MyConfig()

    # Restore parameters from a previous experiment
    if args.config_json is not None:
        config = load_config_json(config, args.config_json)

    # Apply command-line overrides
    config = load_parser(config, args=remaining)

    config.init_dependent_config()

    trainer = SegTrainer(config)

    if config.task == 'train':
        trainer.run(config)
    elif config.task == 'val':
        trainer.validate(config)
    elif config.task == 'predict':
        trainer.predict(config)
    else:
        raise ValueError(f'Unsupported task type: {config.task}.\n')


if __name__ == '__main__':
    main()
