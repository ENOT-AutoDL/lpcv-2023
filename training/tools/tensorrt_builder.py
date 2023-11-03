import argparse
from pathlib import Path
from typing import Union

import tensorrt

DEFAULT_MAX_WORKSPACE_SIZE = 1 * 1024 * 1024 * 1024


class EngineBuilder:
    def __init__(
        self,
        fp16: bool,
        int8: bool,
        max_workspace_size: int = DEFAULT_MAX_WORKSPACE_SIZE,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        fp16 : bool
            Enable FP16 precision.
        int8 : bool
            Enable INT8 precision.
        max_workspace_size : int
            Maximum workspace size in bytes.
            Layer implementations often require a temporary workspace,
            and this parameter limits the maximum size that any layer in the network can use.
            If insufficient workspace is provided,
            it is possible that TensorRT will not be able to find an implementation for a layer.
        verbose : bool
            Enable verbose logging. Default value is False.

        """
        self._fp16 = fp16
        self._int8 = int8
        self._max_workspace_size = max_workspace_size

        severity = tensorrt.Logger.VERBOSE if verbose else tensorrt.Logger.WARNING
        self._logger = tensorrt.Logger(severity)

    def build(self, onnx_path: Union[str, Path], engine_path: Union[str, Path]) -> None:
        """
        Build TensorRT engine from ONNX.

        Parameters
        ----------
        onnx_path : Union[str, Path]
            Path to ONNX network.
        engine_path : Union[str, Path]
            Path to TensorRT engine.

        """
        builder = tensorrt.Builder(self._logger)
        network = self._network(builder, str(onnx_path))
        config = self._config(builder)

        engine = builder.build_engine(network, config)
        with open(file=str(engine_path), mode="wb") as engine_file:
            engine_file.write(engine.serialize())

    def _network(self, builder: tensorrt.Builder, onnx_path: str) -> tensorrt.INetworkDefinition:
        network = builder.create_network(1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        onnx_parser = tensorrt.OnnxParser(network, self._logger)
        onnx_parser.parse_from_file(onnx_path)
        return network

    def _config(self, builder: tensorrt.Builder) -> tensorrt.IBuilderConfig:
        config = builder.create_builder_config()
        config.max_workspace_size = self._max_workspace_size

        if self._fp16:
            config.set_flag(tensorrt.BuilderFlag.FP16)
        if self._int8:
            config.set_flag(tensorrt.BuilderFlag.INT8)

        profile = builder.create_optimization_profile()
        config.add_optimization_profile(profile)

        return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument("--onnx", help="ONNX path", type=str, required=True)
    parser.add_argument("--engine", help="TensorRT engine path", type=str, required=True)
    parser.add_argument("--fp16", help="FP16 precision", action='store_true', default=True)
    parser.add_argument("--int8", help="INT8 precision", action='store_true', default=False)
    args = parser.parse_args()

    builder = EngineBuilder(fp16=args.fp16, int8=args.int8)
    builder.build(onnx_path=args.onnx, engine_path=args.engine)
