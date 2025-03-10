import os, sys, torch
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from configs import MyConfig, load_parser
from models import get_model

import warnings
warnings.filterwarnings("ignore")


class Exporter:
    def __init__(self, config):
        config.use_aux = False
        config.use_detail_head = False

        self.load_ckpt_path = config.load_ckpt_path
        self.export_format = config.export_format
        self.export_size = config.export_size
        self.onnx_opset = config.onnx_opset
        self.export_path = config.export_name + f'.{config.export_format}'
        self.config = config

        self.model = get_model(config)
        self.load_ckpt()

    def load_ckpt(self):
        if not self.load_ckpt_path:     # when set to None
            pass
        elif os.path.isfile(self.load_ckpt_path):
            checkpoint = torch.load(self.load_ckpt_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.model.eval()

            print(f'Loading checkpoint: {self.load_ckpt_path} successfully.\n')
            del checkpoint
        else:
            raise RuntimeError

    def export(self):
        print('\n=========Export=========')
        print(f'Model: {self.config.model}\nEncoder: {self.config.encoder}\nDecoder: {self.config.decoder}')
        print(f'Export Size (H, W): {self.export_size}')
        print(f'Export Format: {self.export_format}')

        if self.export_format == 'onnx':
            from models.modules import replace_adaptive_avg_pool
            self.model = replace_adaptive_avg_pool(self.model)

            self.export_onnx()
            print('\nExporting Finished.\n')

        else:
            raise NotImplementedError

    def export_onnx(self, image=None):
        image = torch.rand(1, 3, *self.export_size) if not image else image
        torch.onnx.export(self.model, image, self.export_path, opset_version=self.onnx_opset, 
                            input_names=['input'], output_names=['output'])


if __name__ == '__main__':
    config = MyConfig()
    config = load_parser(config)
    config.load_ckpt_path = None    # None if you do not have a ckpt to load
    config.init_dependent_config()

    try:
        exporter = Exporter(config)
        exporter.export()
    except Exception as e:
        print(f'\nUnable to export PyTorch model {config.model} to {config.export_format} due to: {e}')