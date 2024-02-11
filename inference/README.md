# LPCV 2023 Challenge - ENOT Evaluation

This repository should be cloned to a Jetson Nano for further inference. Trained model in .onnx format from `training` directory should be moved to this directory.

## Requirements
To setup the environment, refer to [this](https://github.com/lpcvai/23LPCVC_Segmentation_Track-Sample_Solution) repository

## Build TensorRT engine
For model evaluation we use TensorRT for fast inference on device. For building TensorRT engine you can use script `tensorrt/build_engine.sh` where you should write pathes for input `model.onnx` and output engine `model.plan`.

To build engine use the following command:
```bash
bash tensorrt/build_engine.sh
```
Building an engine takes about 15 to 20 minutes. After that move builded engine to solution directory.

## Evaluation
For evaluation model you should create .pyz archive and move it to evaluation directory. for creating .pyz archive you can use the following command:
```bash
bash create_pyz.sh
```
that creates solution.pyz archive.

Then go to evaluation directory, write in `evaluation.bash` pathes to directories with predicted masks and ground truth masks and start evaluation using command:
```bash
bash evaluation.bash
```

To contact us, please visit https://enot.ai or email us at enot@enot.ai.
