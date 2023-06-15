import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from configs import MyConfig
from models import get_model


def cal_model_params(config, imgw=1024, imgh=512):
    model = get_model(config)
    print(f'\nModel: {config.model}\nEncoder: {config.encoder}\nDecoder: {config.decoder}')

    try:
        from ptflops import get_model_complexity_info
        model.eval()
        '''
        Notice that ptflops doesn't take into account torch.nn.functional.* operations.
        If you want to get correct macs result, you need to modify the modules like 
        torch.nn.functional.interpolate to torch.nn.Upsample.
        '''
        _, params = get_model_complexity_info(model, (3, imgh, imgw), as_strings=True, 
                                                print_per_layer_stat=False, verbose=False)
        print(f'Number of parameters: {params}\n')
    except:
        import numpy as np
        params = np.sum([p.numel() for p in model.parameters()])
        print(f'Number of parameters: {params / 1e6:.2f}M\n')


if __name__ == '__main__':
    config = MyConfig()
    
    cal_model_params(config)