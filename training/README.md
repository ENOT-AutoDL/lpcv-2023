# LPCV 2023 Challenge - ENOT Training

## Requirements
To setup environment, install required packages:
```bash
pip install -r requirements.txt
```

## Dataset
Data for training can be downloaded from [here](https://drive.google.com/file/d/1-xqUeSSfDcZFw-rjk28NiZARuN-6sFj8/view?usp=sharing).
Just download this .zip archive and unzip in this directory using command:
```bash
unzip data.zip
``` 

## Training

To reproduce our results, you should train a model with default configuration, specified in `configs/lpcv/pidnet_small_lpcv.yaml`. For training model use the following command:
```bash
bash scripts/train.sh
```

## Validation

To validate your model, you can use the following command:
```bash
bash scripts/val.sh
```

## Converting to ONNX 

For converting model to ONNX, use the following command:
```bash
bash scripts/export_onnx.sh
```

When you will get the .onnx of trained model, you should use [lpcv-2023-inference](https://github.com/LPCV-org/lpcv-2023-inference) repository for evaluating your model on Jetson Nano.

To contact us, please visit https://enot.ai or email us at enot@enot.ai.
