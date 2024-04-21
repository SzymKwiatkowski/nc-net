"""Module providing utilities to model exporting and assessment"""
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


def get_size(model_file_path: Path) -> str:
    """
    :param model_file_path: path pointing to exported model
    :return: size of model
    """
    model_file_path = Path(model_file_path)
    size = model_file_path.stat().st_size

    size_str = None

    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            size_str = f"{size:3.1f} {x}"
            break
        size /= 1024.0

    return "Couldn't specify size!" if size_str is None else size_str


def sparsity(model) -> float:
    """ Calculates global model sparsity. 
    :rtype: sparsity percentage
    """
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def measure_latency(model, input_size: int, dtype=torch.float32, iterations: int = 10000) -> float:
    """

    :param model: pytorch model instance
    :param input_size: count of input features
    :param dtype: dtype of input data
    :param iterations: iterations to measure latency
    :return: average latency (in ms)
    """
    infer_times = []

    for __ in range(iterations):
        input_data = torch.randn(1, 1, input_size, dtype=dtype, device=model.device)
        inference_start = perf_counter()
        __ = model(input_data)
        inference_time = perf_counter() - inference_start
        infer_times.append(inference_time)

        return np.mean(infer_times)


def export_onnx(model, input_size: int, dtype=torch.float32, onnx_model_name: str = 'model.onnx'):
    """
    :param model: pytorch model instance
    :param input_size: count of input features
    :param dtype: dtype of input data
    :param onnx_model_name: name of onnx model
    """
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
