import torch
import argparse
from pathlib import Path
import onnx

from models.model import ControllerModel
from utils.model_utils import export_onnx


def export_model(args):
    model_path = Path(args.model_path)
    torch_model = ControllerModel.load_from_checkpoint(model_path)
    onnx_export_name = args.export_name
    extraction_points = args.extraction_points_count
    input_size = (extraction_points+1) * 8
    # dummy_input = torch.randn(1, input_size)
    #
    # torch.onnx.export(torch_model, dummy_input, onnx_export_name, input_names=['input'],
    #                   output_names=['output'], export_params=True)
    export_onnx(torch_model, input_size=input_size, onnx_model_name=onnx_export_name)

    onnx_model = onnx.load(onnx_export_name)
    onnx.checker.check_model(onnx_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-m', '--model-path', action='store', default='torch_model.pth')
    parser.add_argument('-n', '--export-name', action='store', default='model.onnx')
    parser.add_argument('-d', '--data-path', action='store', default='../data/sample_data.csv')
    parser.add_argument('-ep', '--extraction-points-count', action='store', default=20,
                        type=int, help='Specified count of points from trajectory to be used')
    parsed_args = parser.parse_args()
    export_model(parsed_args)