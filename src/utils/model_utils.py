from pathlib import Path
from time import perf_counter

import numpy as np
import torch


def get_size(model_file_path: Path) -> str:
    """ Calculates model file size. """
    model_file_path = Path(model_file_path)
    size = model_file_path.stat().st_size

    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3.1f %s" % (size, x)
        size /= 1024.0


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sparsity(model):
    """ Calculates global model sparsity. """
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def measure_latency(model, input_size: int, dtype=torch.float32, iterations: int = 10000) -> float:
    infer_times = []

    for __ in range(iterations):
        input_data = torch.randn(1, 1, input_size, dtype=dtype, device=model.device)
        inference_start = perf_counter()
        __ = model(input_data)
        inference_time = perf_counter() - inference_start
        infer_times.append(inference_time)

        return np.mean(infer_times)


def export_onnx(model, input_size: int, dtype=torch.float32, onnx_model_name: str = 'model.onnx'):
    model.eval()
    input_data = torch.randn(1, 1, input_size, dtype=dtype, device=model.device)

    torch.onnx.export(
        model,
        input_data,  # model input (or a tuple for multiple inputs)
        onnx_model_name,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=16,  # the ONNX version to export the model to
        input_names=['image'],
        output_names=['output'],
        do_constant_folding=False
    )
