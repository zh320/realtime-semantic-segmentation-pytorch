import sys, time, torch
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from configs import MyConfig, load_parser
from models import get_model, model_hub


def test_model_speed(config, mode='cuda', ratio=0.5, imgw=2048, imgh=1024, iterations=None):
    if mode == 'cuda':
        # Codes are based on 
        # https://github.com/ydhongHIT/DDRNet/blob/main/segmentation/DDRNet_23_slim_eval_speed.py

        if ratio != 1.0:
            assert ratio > 0, 'Ratio should be larger than 0.\n'
            imgw = int(imgw * ratio)
            imgh = int(imgh * ratio)

        device = torch.device('cuda')
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True

        model = get_model(config)
        model.eval()
        model.to(device)
        print('\n=========Speed Testing=========')
        print(f'Model: {config.model}\nEncoder: {config.encoder}\nDecoder: {config.decoder}')
        print(f'Size (W, H): {imgw}, {imgh}')

        input = torch.randn(1, 3, imgh, imgw).cuda()
        with torch.no_grad():
            for _ in range(10):
                model(input)

            if iterations is None:
                elapsed_time = 0
                iterations = 100
                while elapsed_time < 1:
                    torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    t_start = time.time()
                    for _ in range(iterations):
                        model(input)
                    torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    elapsed_time = time.time() - t_start
                    iterations *= 2
                FPS = iterations / elapsed_time
                iterations = int(FPS * 6)

            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t_start = time.time()
            for _ in range(iterations):
                model(input)
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            elapsed_time = time.time() - t_start
            latency = elapsed_time / iterations * 1000
        torch.cuda.empty_cache()
        FPS = 1000 / latency

    elif mode == 'cpu':
        import numpy as np
        import onnxruntime as ort
        from tools.export import Exporter

        try:
            config.export_name = f'{config.model}_dummy'
            exporter = Exporter(config)
            exporter.export()
        except Exception as e:
            print(f'\nUnable to export PyTorch model {config.model} to ONNX due to: {e}')
            return -1

        load_onnx_path = f'{config.model}_dummy.onnx' if not config.load_onnx_path else config.load_onnx_path

        print('\n=========Speed Testing=========')
        print(f'Model: {config.model}\nEncoder: {config.encoder}\nDecoder: {config.decoder}')
        print(f'Size (H, W): {config.export_size}')

        session = ort.InferenceSession(load_onnx_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape

        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        print('\nRunning CPU warmup...')
        for _ in range(10):
            session.run(None, {input_name: dummy_input})

        num_iterations = iterations if iterations else 100
        print('Start speed testing on CPU using ONNX runtime...')
        start_time = time.time()
        for _ in range(num_iterations):
            session.run(None, {input_name: dummy_input})

        end_time = time.time()
        FPS = num_iterations / (end_time - start_time)

    else:
        raise NotImplementedError

    print(f'FPS: {FPS}\n')
    return FPS


if __name__ == '__main__':
    mode = 'cpu'
    test_all_model = False

    config = MyConfig()
    config = load_parser(config)
    config.use_aux = False
    config.use_detail_head = False
    config.load_ckpt_path = None        # None if you do not have a ckpt to load and export to ONNX
    config.init_dependent_config()

    with open(f'{mode}_perf.txt', 'w') as f:
        f.write('model\t\tFPS\n')

    if test_all_model:
        for model_name in sorted(model_hub.keys()):
            config.model = model_name

            fps = test_model_speed(config, mode=mode)
            with open(f'{mode}_perf.txt', 'a+') as f:
                f.write(f'{config.model}\t\t{fps:.2f}\n')

    elif config.model in model_hub.keys():
        fps = test_model_speed(config, mode=mode)
        with open(f'{mode}_perf.txt', 'a+') as f:
            f.write(f'{config.model}\t\t{fps:.2f}\n')

    else:
        raise ValueError(f'Unsupported model: {config.model}\n')