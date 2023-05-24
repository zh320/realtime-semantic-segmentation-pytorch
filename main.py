from core import SegTrainer
from configs import MyConfig

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    config = MyConfig()
    
    config.init_dependent_config()
    
    trainer = SegTrainer(config)
    
    if config.is_testing:
        trainer.predict(config)
    else:
        trainer.run(config)