import argparse
import tempfile

import _init_paths  # do not remove
import onnx
import onnxruntime
import onnxsim
import torch
import torch.optim

import models


def parse_args():
    parser = argparse.ArgumentParser(description="Export PyTorch checkpoint to ONNX")
    parser.add_argument("--checkpoint", help="weights path", type=str, required=True)
    parser.add_argument("--onnx-path", help="ONNX output path", type=str, required=True)
    parser.add_argument("--input-shape", help="model input shape", type=int, nargs="+", required=True)
    parser.add_argument("--opset", help="ONNX opset", type=int, required=True)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    model = models.pidnet.get_seg_model(
        model_name="pidnet_s",
        num_classes=14,
        checkpoint_path=args.checkpoint,
        aux_heads=True,
    )
    model.eval()

    with tempfile.TemporaryDirectory() as tempdir:
        torch.onnx.export(
            model=model,
            args=torch.rand(*args.input_shape),
            f=f"{tempdir}/model.onnx",
            input_names=["input"],
            output_names=["unused_output_0", "output", "unused_output_1"],
            opset_version=args.opset,
        )

        # apply onnxsim:
        model, _ = onnxsim.simplify(
            model=f"{tempdir}/model.onnx",
            unused_output=["unused_output_0", "unused_output_1"],
        )
        onnx.save(model, f"{tempdir}/model.onnx")

        # apply onnxruntime:
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess_options.optimized_model_filepath = args.onnx_path
        onnxruntime.InferenceSession(f"{tempdir}/model.onnx", sess_options)


if __name__ == "__main__":
    main()
