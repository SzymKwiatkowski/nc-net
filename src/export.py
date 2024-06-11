"""Script for exporting pytorch models to ONNX format."""
import argparse
from pathlib import Path
import onnx

from models.model import ControllerModel
from utils.model_utils import export_onnx, measure_latency, get_size
from utils.resource_manager import ResourceManager


# pylint: disable=E1120
def export_model(args) -> None:
    """
    Export model to ONNX format
    :rtype: None
    """
    model_path = Path(args.model_path)
    torch_model = ControllerModel.load_from_checkpoint(checkpoint_path=model_path)
    onnx_export_name = args.export_name
    extraction_points = args.extraction_points_count
    input_size = args.input_size
    assessment = args.assessment
    if input_size == 0 or input_size is None:
        input_size = (extraction_points+1) * len(ResourceManager.get_regex_point_position_patterns_short())

    export_onnx(torch_model, input_size=input_size, onnx_model_name=onnx_export_name)

    onnx_model = onnx.load(onnx_export_name)
    onnx.checker.check_model(onnx_model)

    if assessment:
        print("Model size: ", get_size(onnx_export_name))
        print("Average infer time: ", measure_latency(torch_model, input_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='export',
        description='Export model in ONNX format',
        epilog='')
    parser.add_argument('-m', '--model-path', action='store', default='torch_model.pth')
    parser.add_argument('-n', '--export-name', action='store', default='model.onnx')
    parser.add_argument('-ep', '--extraction-points-count', action='store', default=10,
                        type=int, help='Specified count of points from trajectory to be used')
    parser.add_argument('-ip', '--input-size', action='store', type=int,
                        help='Specified input size if passed to script automatically overrides' +
                             ' extraction points (whole pose is taken into account)')
    parser.add_argument('-a', '--assessment', action='store', type=bool, default=False,
                        help='Specifies whether to assess model performance')
    parsed_args = parser.parse_args()
    export_model(parsed_args)
