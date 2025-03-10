from core import SegTrainer
from configs import MyConfig, load_parser

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    config = MyConfig()

    config.init_dependent_config()

    # If you want to use command-line arguments, please uncomment the following line
    # config = load_parser(config)

    trainer = SegTrainer(config)

    if config.task == 'train':
        trainer.run(config)
    elif config.task == 'val':
        trainer.validate(config)
    elif config.task == 'predict':
        trainer.predict(config)
    else:    
        raise ValueError(f'Unsupported task type: {config.task}.\n')